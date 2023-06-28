## Compute foliar moisture content of a 100,000 km2 area of tree cover reflectance data based on the top-left coordinates of an area

# example usage of this module: compute_fmc.py -x 1026330.0 -y -3181590.0

import datacube
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import sys
sys.path.insert(1, '../Tools/')
from dea_tools.datahandling import load_ard # https://github.com/GeoscienceAustralia/dea-notebooks/tree/develop/Tools/
import pickle5 as pickle
import time
import argparse
from pathlib import Path
    
def run_modelling(x_left,y_top):    
    t0 = time.time()

    # Create datacube instance
    dc = datacube.Datacube(app='fmc_nsw')

    # Load mask of tree cover which is reprojected to Albers/20m
    trees = xr.open_dataarray('data/nonveg_mask_nve_pct_nsw.tiff',
                                    engine='rasterio',chunks='auto').drop('band').squeeze('band')
    # Create time dimension which includes every day since 2015 (ie whole Sentinel data period)
    dates = pd.date_range('2015-01-01',datetime.date.today(), freq="D")
    # Add time dimension to tree data mask, in a new xr dataset
    trees = xr.Dataset({'trees': trees, 'time': dates})
    trees.attrs['crs'] = None
    trees.attrs['crs'] = 'EPSG:3577'

    # Subset the trees raster to the array of interest
    arr_size = 5000
    arr_pix_size = int(arr_size * 20)
    subs = trees.sel(x=slice(x_left,(x_left+arr_pix_size-1)),y=slice(y_top,(y_top-arr_pix_size+1)))
    
    # Define bands of Sentinel-2 reflectance to retrieve
    bands = ['nbart_red','nbart_green','nbart_blue','nbart_red_edge_1','nbart_red_edge_2',
             'nbart_red_edge_3','nbart_nir_1','nbart_nir_2','nbart_swir_2','nbart_swir_3']

    ## Get Sentinel-2 reflectance data from DEA 
    # Query the datacube using the 'like' attribute of load_ard, to take the spatial and temporal resolution of the trees dataset
    s2_cube = load_ard(dc=dc, 
                   products=['ga_s2am_ard_3','ga_s2bm_ard_3'],
                   like=subs, 
                   measurements=bands,
                   group_by='solar_day', 
                   dask_chunks = {'time':1},
                   cloud_mask='fmask')

    # Add NDVI and NDII
    s2_cube['ndii'] = ((s2_cube.nbart_nir_1-s2_cube.nbart_swir_2)/(s2_cube.nbart_nir_1+s2_cube.nbart_swir_2))
    s2_cube['ndvi'] = ((s2_cube.nbart_nir_1-s2_cube.nbart_red)/(s2_cube.nbart_nir_1+s2_cube.nbart_red))

    # Add some metadata to array
    s2_cube.attrs['site'] = (x_left,y_top)
    s2_cube.time.attrs = {}
    
    # Define predictors to pass to random forest model from the variables in the array
    rf_predictors = ['ndvi','ndii','nbart_red','nbart_green','nbart_blue','nbart_red_edge_1','nbart_red_edge_2',
             'nbart_red_edge_3','nbart_nir_1','nbart_nir_2','nbart_swir_2','nbart_swir_3']

    # Load the trained random forest model from disk
    with open('rf_xr_s2fmc_forest.pickle', 'rb') as handle:
        rf = pickle.load(handle)
    rf_predictors = ['ndvi','ndii']+bands
    rf.set_params(n_jobs=-1)
    rf.set_params(reshapes='variable')

    t1 = time.time()
    print('Array coords:',x_left,y_top)
    print((t1-t0)/60,'mins to prepare data')

    # Run model on each timestep of dataset
    for t in s2_cube.time.data:
        path = Path(f'output/{str(int(x_left))}_{str(int(y_top))}_{str(t)[:10]}.nc')
        if path.is_file():
            continue 
        t0 = time.time()
        data = s2_cube.sel(time=t) # selects timestep
        data = data.chunk('auto') # sets the chunksize of the array to be spread across dask workers
        data = data[rf_predictors].to_array().stack(index=['y','x']).transpose() # reshapes the dataset to an array to suit random forest
        data = data.where(np.isfinite(data), 0) # sets all nan kind of data to 0 to suit random forest
        data = rf.predict(data) # runs random forest
        data = data.unstack(dim=['index']) # reverses the reshaping back to 2 dimensions (x and y)
        data = data.where(np.isfinite(s2_cube.sel(time=t)['ndii'])) # reverses the setting of 0 to nan; applies mask from reflectance data
        data = data.where((subs.trees==1)) # masks pixels which are not tree cover
        data = data.assign_coords({'time':t}).expand_dims('time') # assign the time dimension back to array
        data.to_netcdf(path) # saves out the timestep
        t1 = time.time()
        print('Tile/job params:',x_left,y_top)
        print((t1-t0)/60,'mins to compute timestep',str(t))

if __name__ == "__main__":
    # Parse the coordinates arguments given to this script
    parser = argparse.ArgumentParser(description="""Model FMC of sections of tree cover raster""")
    parser.add_argument('-x', '--x_left', type=float, required=True, help="Specifies the left x coordinate for a section")
    parser.add_argument('-y', '--y_top', type=float, required=True, help="Specifies the top y coordinate for a section")
    args = parser.parse_args()
    # Run the central function which loads reflectance and runs the FMC model (a random forest regression)
    run_modelling(args.x_left,args.y_top)
