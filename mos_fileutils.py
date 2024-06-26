import os
import sys
import datetime
import numpy as np
import pyproj
import eccodes as ecc
import fsspec
import pandas as pd


def read_grib(gribfile, read_coordinates=False):
    """Read first message from grib file and return content.
    List of coordinates is only returned on request, as it's quite
    slow to generate.
    """
    forecasttime = []
    values = []

    print(f"Reading file {gribfile}")
    wrk_gribfile = get_local_file(gribfile)

    lons = []
    lats = []

    with open(wrk_gribfile, "rb") as fp:
        # print("Reading {}".format(gribfile))

        while True:
            try:
                gh = ecc.codes_grib_new_from_file(fp)
            except ecc.WrongLengthError as e:
                #print(e)
                file_stats = os.stat(wrk_gribfile)
                print("Size of {}: {}".format(wrk_gribfile, file_stats.st_size))
                sys.exit(1)

            if gh is None:
                break

            ni = ecc.codes_get_long(gh, "Nx")
            nj = ecc.codes_get_long(gh, "Ny")
            dataDate = ecc.codes_get_long(gh, "dataDate")
            dataTime = ecc.codes_get_long(gh, "dataTime")
            lat_first = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
            lon_first = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")
            lat_last = ecc.codes_get_double(gh, "latitudeOfLastGridPointInDegrees")
            lon_last = ecc.codes_get_double(gh, "longitudeOfLastGridPointInDegrees")
            forecastTime = ecc.codes_get_long(gh, "endStep")
            analysistime = datetime.datetime.strptime(
                "{}.{:04d}".format(dataDate, dataTime), "%Y%m%d.%H%M"
            )

            ftime = analysistime + datetime.timedelta(hours=forecastTime)
            forecasttime.append(ftime)

            tempvals = ecc.codes_get_values(gh).reshape(nj, ni)
            values.append(tempvals)
            #print(values)

            if read_coordinates and len(lons) == 0:
                projstr = get_projstr(gh)
                proj_to_ll = pyproj.Transformer.from_crs(projstr, "epsg:4326")

                if projstr.startswith("+proj=lcc") or projstr.startswith("+proj=stere"):
                    
                    di = ecc.codes_get_double(gh, "DxInMetres")
                    dj = ecc.codes_get_double(gh, "DyInMetres")

                    for j in range(nj):
                        y = j * dj
                        for i in range(ni):
                            x = i * di

                            lat, lon = proj_to_ll.transform(x, y)
                            lons.append(lon)
                            lats.append(lat)

                elif projstr.startswith("+proj=latlong"):
                    # Get latitude and longitude of first grid point
                    #lat_first = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
                    #lon_first = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")
                    # Calculate grid spacing based on differences between adjacent points
                    lat_step = ecc.codes_get_double(gh, "iDirectionIncrementInDegrees")
                    lon_step = ecc.codes_get_double(gh, "jDirectionIncrementInDegrees")

                    # Calculate latitudes and longitudes for each grid point
                    for j in range(nj):
                        for i in range(ni):
                            lat = lat_first - j * lat_step
                            lon = lon_first + i * lon_step

                            # Transform lat/lon to EPSG:4326 (WGS 84)
                            #lon, lat = proj_to_ll.transform(lon, lat)

                            lons.append(lon)
                            lats.append(lat)


                else:
                    print("Unsupported projection: {}".format(projstr))
                    sys.exit(1)

        if read_coordinates == False and len(values) == 1:
            return (
                None,
                None,
                np.asarray(values).reshape(nj, ni),
                analysistime,
                forecasttime,
            )
        elif read_coordinates == False and len(values) > 1:
            return None, None, np.asarray(values), analysistime, forecasttime
        else:
            return (
                np.asarray(lons).reshape(nj, ni),
                np.asarray(lats).reshape(nj, ni),
                np.asarray(values),
                analysistime,
                forecasttime,
            )

def get_shapeofearth(gh):
    """Return correct shape of earth sphere / ellipsoid in proj string format.
    Source data is grib2 definition.
    """

    shape = ecc.codes_get_long(gh, "shapeOfTheEarth")

    if shape == 1:
        v = ecc.codes_get_long(gh, "scaledValueOfRadiusOfSphericalEarth")
        s = ecc.codes_get_long(gh, "scaleFactorOfRadiusOfSphericalEarth")
        return "+R={}".format(v * pow(10, s))

    if shape == 5:
        return "+ellps=WGS84"

    if shape == 6:
        return "+R=6371229.0"

def get_falsings(projstr, lon0, lat0):
    """Get east and north falsing for projected grib data"""

    ll_to_projected = pyproj.Transformer.from_crs("epsg:4326", projstr)
    return ll_to_projected.transform(lat0, lon0)

def get_projstr(gh):
    """Create proj4 type projection string from grib metadata" """

    projstr = None
    proj = ecc.codes_get_string(gh, "gridType")
    first_lat = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
    first_lon = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")

    if proj == "polar_stereographic":
        projstr = "+proj=stere +lat_0=90 +lat_ts={} +lon_0={} {} +no_defs".format(
            ecc.codes_get_double(gh, "LaDInDegrees"),
            ecc.codes_get_double(gh, "orientationOfTheGridInDegrees"),
            get_shapeofearth(gh),
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    elif proj == "lambert":
        projstr = (
            "+proj=lcc +lat_0={} +lat_1={} +lat_2={} +lon_0={} {} +no_defs".format(
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin2InDegrees"),
                ecc.codes_get_double(gh, "LoVInDegrees"),
                get_shapeofearth(gh),
            )
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    #elif proj == "regular_ll":
    #    projstr = "+proj=ob_tran +o_proj=latlon +o_lon_p={} +o_lat_p={} +lon_0={} {} +no_defs".format(
    #        ecc.codes_get_double(gh, "angleOfRotationInDegrees"),
    #        ecc.codes_get_double(gh, "latitudeOfSouthernPoleInDegrees"),
    #        ecc.codes_get_double(gh, "longitudeOfSouthernPoleInDegrees"),
    #        get_shapeofearth(gh),
    #    )

        
    elif proj == "regular_ll":
        #projstr = "+proj=latlong +ellps={} +no_defs".format(get_shapeofearth(gh))
        #projstr = "+proj=latlong +ellps=WGS84 +no_defs"
        #projstr = "+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs"
        projstr = "+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs"


    else:
        print("Unsupported projection: {}".format(proj))
        sys.exit(1)

    return projstr


def get_local_file(grib_file):
    if not grib_file.startswith("s3://"):
        return grib_file

    uri = "simplecache::{}".format(grib_file)

    return fsspec.open_local(
        uri,
        mode="rb",
        s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}},
    )

def read_mos_csv(mos_file):
    mosfile = get_local_file(mos_file)
    mf = pd.read_csv(mosfile)
    return mf  


def write_grib_message(fpout, analysistime, forecasttime, data, template):
    with open(get_local_file(template)) as fpin:
        for tdata in data:
            #print(tdata.shape)
            h = ecc.codes_grib_new_from_file(fpin)
            #print(h)

            assert h is not None, "Template file length is less than output data length"
            ecc.codes_set_values(h, tdata.flatten())
            ecc.codes_write(h, fpout)
            ecc.codes_release(h)


def write_grib(outputf, analysistime, forecasttime, data, template):
    if outputf.startswith("s3://"):
        openfile = fsspec.open(
            "simplecache::{}".format(outputf),
            "wb",
            s3={
                "anon": False,
                "key": os.environ["S3_ACCESS_KEY_ID"],
                "secret": os.environ["S3_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
            },
        )
        with openfile as fpout:
            write_grib_message(fpout, analysistime, forecasttime, data, template)
    else:
        with open(outputf, "wb") as fpout:
            write_grib_message(fpout, analysistime, forecasttime, data, template)

    print(f"Wrote file {outputf}")