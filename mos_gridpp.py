# test gridpp gridding to mos

# read input (mos-forecasts in csv, ecmwf background grib-files) data from s3://routines-data/mos
# needs following functions:
# read/write grib
# create locations variables needed in gridpp
# run for t and td
# run gridding for all the leadtimes
# plot output

import gridpp
from mos_fileutils import write_grib, read_grib, read_mos_csv
import numpy as np
import eccodes as ecc
import sys
#import pyproj
import requests
import datetime
import argparse
import pandas as pd
#from scipy.interpolate import RegularGridInterpolator
import fsspec
import os
import time
import copy
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
#import rioxarray
#from flatten_json import flatten
#import gzip
#from multiprocessing import Process, Queue


#warnings.filterwarnings("ignore")

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mos_csv", action="store", type=str, required=True)
    #parser.add_argument("--obs_ec_bg", action="store", type=str, required=True)
    parser.add_argument("--ec_bg", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    #parser.add_argument("--plot", action="store_true", default=False)
    #parser.add_argument("--disable_multiprocessing", action="store_true", default=False)

    args = parser.parse_args()

    allowed_params = ["temperature", "dewpoint"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args

def read_grid(args):
    # """Top function to read "all" gridded data"""
    # one grib file vontains one leadtime with 2t, 2d, z and lsm info

    lons, lats, vals, analysistime, forecasttime = read_grib(args.ec_bg, True)
    t2m = vals[0]
    tdew = vals[1]
    topo = vals[2]
    topo = topo / 9.81
    lc = vals[3]
    if args.parameter=="temperature":
        par_val = t2m 
    elif args.parameter=="dewpoint":    
        par_val = tdew

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, par_val,  analysistime, forecasttime, lc, topo


def interpolate_single_time(grid, background, points, obs, obs_to_background_variance_ratio, pobs, structure, max_points, idx, q):
    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
        grid,
        background[idx],
        points[idx],
        obs[idx]["bias"].to_numpy(),
        obs_to_background_variance_ratio[idx],
        pobs[idx],
        structure,
        max_points,
    )

    print(
        "step {} min grid: {:.1f} max grid: {:.1f}".format(
            idx, np.amin(tmp_output), np.amax(tmp_output)
        )
    )

    if q is not None:
        # return index and output, so that the results can
        # later be sorted correctly
        q.put((idx, tmp_output))
    else:
        return tmp_output


def interpolate(grid, points, background, obs, args):
    # interpolate(grid, obs, background, args, lc)
    """Perform optimal interpolation"""

    output = []
    # create a mask to restrict the modifications only to land area (where lc = 1)
    #lc0 = np.logical_not(lc).astype(int)

    # Interpolate background data to observation points
    pobs = gridpp.nearest(grid, points, background)
    obs_to_background_variance_ratio = np.full(points.size(), 0.1)
    # Barnes structure function with horizontal decorrelation length 30km, vertical decorrelation length 200m
    structure = gridpp.BarnesStructure(30000, 200, 0.5)
    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20
    output = gridpp.optimal_interpolation(
        grid,
        background,
        points,
        obs,
        obs_to_background_variance_ratio,
        pobs,
        structure,
        max_points,
    )
    return output

    """

    pobs = []
    obs_to_background_variance_ratio = []
    for i in range(0, len(obs)):
        # interpolate background to obs points
        pobs.append(gridpp.nearest(grid, points[i], background[i]))
        obs_to_background_variance_ratio.append(np.full(points[i].size(), 0.1))
        
    # Barnes structure function with horizontal decorrelation length 30km, vertical decorrelation length 200m
    structure = gridpp.BarnesStructure(30000, 200, 0.5)

    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20

    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    #obs_to_background_variance_ratio = np.full(points.size(), 0.1)

    if args.disable_multiprocessing:
        output = [
            interpolate_single_time(
                grid,
                background,
                points,
                obs,
                obs_to_background_variance_ratio,
                pobs,
                structure,
                max_points,
                x,
                None,
            )
            for x in range(len(obs))
        ]

    else:
        q = Queue()
        processes = []
        outputd = {}

        for i in range(len(obs)):
            processes.append(
                Process(
                    target=interpolate_single_time,
                    args=(
                        grid,
                        background,
                        points,
                        obs,
                        obs_to_background_variance_ratio,
                        pobs,
                        structure,
                        max_points,
                        i,
                        q,
                    ),
                )
            )
            processes[-1].start()

        for p in processes:
            # get return values from queue
            # they might be in any order (non-consecutive)
            ret = q.get()
            outputd[ret[0]] = ret[1]

        for p in processes:
            p.join()

        for i in range(len(obs)):
            # sort return values from 0 to 8
            output.append(outputd[i])

    return output
    """

def main():
    args = parse_command_line()

    print("Reading EC data for", args.parameter )
    # read in the parameter which is forecasted
    # background contains mnwc values for different leadtimes
    st = time.time()
    grid, lons, lats, background, analysistime, forecasttime, lc, topo = read_grid(args)


    # read the mos forecasts from csv-files 
    #mos = pd.read_csv(args.mos_csv)
    mos = read_mos_csv(args.mos_csv) 
    #print(mos.head(5))
    point_lat = mos['latitude'].to_numpy()
    point_lon = mos['longitude'].to_numpy()
    if args.parameter == "temperature":
        point_value = mos['temperature'].to_numpy() + 273.15
    elif args.parameter == "dewpoint":
        point_value = mos['dewpoint'].to_numpy() + 273.15

    # define lsm for obs points using model lsm
    # for interpolating info from lsm, create gridpp.Points object
    points1 = gridpp.Points(
        mos["latitude"].to_numpy(),
        mos["longitude"].to_numpy(),
    )
    # interpolate nearest land sea mask values from grid to obs points (NWP data used, since there's no lsm info from obs stations available)
    mos_lsm = gridpp.nearest(grid, points1, lc)

    points = gridpp.Points(
        mos["latitude"].to_numpy(),
        mos["longitude"].to_numpy(),
        mos["elevation"].to_numpy(),
        mos_lsm,
    )

    # create "zero" background for interpolating the bias
    # background0 = copy.copy(background)
    # background0[background0 != 0] = 0

    et = time.time()
    timedif = et - st
    print(
        "Reading input data for", args.parameter, "takes:", round(timedif, 1), "seconds"
    )

    print("analysistime",analysistime)
    
    # Interpolate mos to background grid
    output = interpolate(grid, points, background, point_value, args)
    diff = background - output

    vmin = min(np.amin(output), np.amin(background))
    vmax = max(np.amax(output), np.amax(background))
    vmin1 = np.amin(diff)
    vmax1 = np.amax(diff)
    print(vmin1, vmax1)
    lt1 = np.amin(lats) #44 # np.amin(lats)
    lt2 = np.amax(lats) #48 # np.amax(lats)
    ln1 = np.amin(lons) #6 # np.amin(lons)
    ln2 = np.amax(lons) #15 # np.amax(lons)

    plt.figure(1)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.subplot(1, 3, 1)
    plt.pcolormesh(
        np.asarray(lons),
        np.asarray(lats),
        background,
        cmap="Spectral_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlim(ln1,ln2)#(0, 35)
    plt.ylim(lt1,lt2)#(55, 75)
    cbar = plt.colorbar(
        label="ecmwf background " + args.parameter, orientation="horizontal"
    )

    plt.subplot(1, 3, 2)
    plt.pcolormesh(
        np.asarray(lons),
        np.asarray(lats),
        diff,
        cmap="RdBu_r",
        vmin=(-5),
        vmax=5,
    )
    plt.xlim(ln1,ln2)#(0, 35)
    plt.ylim(lt1,lt2)#(55, 75)
    cbar = plt.colorbar(
        label="diff2 " + args.parameter, orientation="horizontal"
    )

    plt.subplot(1, 3, 3)
    plt.pcolormesh(
        np.asarray(lons),
        np.asarray(lats),
        output,
        cmap="Spectral_r",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlim(ln1,ln2)#(0, 35)
    plt.ylim(lt1,lt2)#(55, 75)
    cbar = plt.colorbar(
        label="output " + args.parameter, orientation="horizontal"
    )

    plt.savefig("diff_grid" + args.parameter + ".png")


    #write_grib(args.output, analysistime, forecasttime, output, args.ec_bg)
    exit()

    bias_obs = []
    for i in range(0, len(obs)):
        tmp_obs = obs[i]
        # interpolate background to obs points
        tmp_bg_point = gridpp.nearest(grid, points[i], background[i])
        tmp_obs['bias'] = tmp_bg_point - tmp_obs['obs_value']
        # interpolate background0 to obs points
        bias_obs.append(tmp_obs)

    diff = interpolate(grid, points, background0, bias_obs, args)
    intt = time.time()
    timedif = intt - ot
    
    print("Interpolating data takes:", round(timedif, 1), "seconds")
    
    # and convert parameter to T-K or RH-0TO1
    output = []
    output_v = []
    output_u = []

    for j in range(0, len(diff)):
        #print(obs[j].shape[0])
        if obs[j].shape[0] == 1: # just one row of obs == missing obs
            tmp_output = background[j]
        else: 
            tmp_output = background[j] - diff[j]
        # Implement simple QC thresholds
        if args.parameter == "r":
            tmp_output = np.clip(tmp_output, 5, 100)  # min RH 5% !
            tmp_output = tmp_output / 100
        elif args.parameter == "uv":
            tmp_output = np.clip(tmp_output, 0, 38)  # max ws same as in oper qc: 38m/s
            # calculate corrected U and V components from correceted ws and original wd
            wd_met = (270-np.arctan2(v_comp[j],u_comp[j])*180/np.pi) % 360 
            tmp_output_u = -tmp_output*np.sin(wd_met*np.pi/180) 
            tmp_output_v = -tmp_output*np.cos(wd_met*np.pi/180) 
        elif args.parameter == "fg":
            tmp_output = np.clip(tmp_output, 0, 50)
        else: # temperature
            tmp_output = tmp_output + 273.15
        
        if args.parameter != "uv":
            output.append(tmp_output)
        elif args.parameter == "uv":
            output.append(tmp_output) # corrected wind speed
            output_v.append(tmp_output_v) 
            output_u.append(tmp_output_u)


    #print("Interpolating forecasts takes:", round(timedif, 1), "seconds")
    #assert len(forecasttime) == len(output)

    if args.parameter != "uv":
        write_grib(args.output, analysistime, forecasttime, output, args.parameter_data)
    elif args.parameter == "uv":
        write_grib(args.output, analysistime, forecasttime, output_u, args.parameter_data)
        write_grib(args.output_v, analysistime, forecasttime, output_v, args.v_component)

if __name__ == "__main__":
    main()