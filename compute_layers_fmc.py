#!/usr/bin/env python
# coding: utf-8

# # Reduce foliar moisture timeseries to a selection of different layers

# In[138]:


import glob
import xarray as xr
import os
from pathlib import Path
import argparse
import rasterio
import rioxarray
import numpy as np


# In[48]:


pth = '/g/data/bz23/IvanK/fmc_rf_nsw/'


# In[142]:


def mean_fmc(ds,x_left, y_top):
    '''Calculate mean of a xarray object across the time dimension'''
    return ds.mean(dim='time')

def std_fmc(ds,x_left, y_top):
    '''Calculate standard deviation of a xarray object across the time dimension'''
    return ds.std(dim='time')

def fifthperc_fmc(ds,x_left, y_top):
    '''Calculate 5th percentile of a xarray object across the time dimension'''
    return ds.quantile(0.05,'time')

def season_5th_fmc(ds,x_left, y_top):
    '''Calculate 5th percentile for each season of a xarray object across the time dimension'''
    return ds.groupby('time.season').quantile(0.05,dim='time')


# In[143]:


def driestq_5th_fmc(ds,x_left, y_top):
    '''Calculate the 5th percentile of the average driest quarter of a xarray object across the time dimension'''
    # Load start date of dry quarter layer for this tile
    dry_quarter = xr.open_dataarray('/g/data/bz23/IvanK/layers_driest_warmest/outputs/rain_day_start_date_dry_quarter_yrmean_numericDOY_20yrs.nc')
    dry_quarter = dry_quarter.rio.write_crs('EPSG:4326')
    dry_quarter = dry_quarter.rio.reproject_match(ds,resampling=rasterio.enums.Resampling.cubic_spline)
    dry_quarter = np.floor(dry_quarter) # rounds down to int - i.e. 91.1 days is on the 91st day

    # Make a 3d mask including data which time is between start of dry quarter and end of it
    in_quarter_ds = []
    for t in ds['time']:
        in_quarter = (dry_quarter <= ds.sel(time=t)['time'].dt.dayofyear) & (
                                    ds.sel(time=t)['time'].dt.dayofyear < dry_quarter+(12*7))
        in_quarter_ds.append(in_quarter.expand_dims('time'))
    in_quarter_ds = xr.concat(in_quarter_ds,dim='time')

    # # Mask out data not in quarter
    ds = ds.where(((dry_quarter <= ds.time.dt.dayofyear) & (ds.time.dt.dayofyear < dry_quarter+(12*7))))
    ds = ds.chunk({'time':-1,'x':'auto','y':'auto'}) #shape for time reduction
    return ds.quantile(0.05,'time')


# In[144]:


def perc_obs_low_fmc(ds,x_left, y_top):
    '''Calculate the proportion of observations which are less than one standard deviation below the mean of a xarray object across the time dimension'''
    # Load the mean and std dev from previous function calls - one at a time
    # thresh = xr.open_dataarray(f'/g/data/bz23/IvanK/fmc_rf_nsw/layers/tiles/fmc_nsw_{x_left}_{y_top}_mean.nc')
    # thresh = thresh - xr.open_dataarray(f'/g/data/bz23/IvanK/fmc_rf_nsw/layers/tiles/fmc_nsw_{x_left}_{y_top}_std.nc')
    
    thresh = xr.open_dataarray(pth+f'/layers/tiles/{int(x_left)}_{int(y_top)}/fmc_nsw_{int(x_left)}_{int(y_top)}_mean.nc')
    thresh = thresh - xr.open_dataarray(pth+f'/layers/tiles/{int(x_left)}_{int(y_top)}/fmc_nsw_{int(x_left)}_{int(y_top)}_std.nc')
    
    # Count time below threshold by making boolean and then summing along time (i.e. counts all True)
    low_fmc = (ds < thresh).sum(dim='time')
    # Count all non null observations along time
    non_nan_n = (~ds.isnull()).sum(dim='time')
    # Calculate percentage of former of later
    return (low_fmc / non_nan_n) * 100


# In[165]:


def timeseries_functions(x_left, y_top):
    '''Apply each of the timeseries reduction functions to tiles of foliar moisture content. Saves files straight to disk; one for each function except the seasonal 5th percentile function.'''
    # Get the directory paths of each tile of FMC across NSW
    # tile_pths = glob.glob('/g/data/bz23/IvanK/fmc_rf_nsw/*_*/')

    # Dictionary of functions to apply to timeseries
    funcs = {
            'mean':mean_fmc,
            'std':std_fmc,
            '5th_perc':fifthperc_fmc,
            'seas_5th_perc':season_5th_fmc,
            'dryq_5th_per':driestq_5th_fmc,
            'perc_obs_below1sd':perc_obs_low_fmc
    }
    #funcs = {'perc_obs_below1sd':perc_obs_low_fmc} ## TEMPORARY - for testing one at a time

    # Load data from all years and files (timepoints) in this tile
    in_path = pth+f'tiles/{int(x_left)}_{int(y_top)}/*/*.nc'
    in_paths = glob.glob(in_path)
    #print(in_paths)
    ds = xr.open_mfdataset(in_path,chunks={'time':-1,'x':'auto','y':'auto'})
    ds = ds.chunk({'time':-1,'x':'auto','y':'auto'})
    # ds = ds.rio.set_crs('EPSG:3577')
    # ds.attrs['crs'] = 'EPSG:3577'
    
    # TEMPORARY - for testing funcs
    #ds = xr.open_mfdataset('data_analy/outputs/lut_rf_version/*_Kentlyn*.nc',chunks='auto')
    #ds = ds.chunk({'time':-1,'x':'auto','y':'auto'})
    
    # Encoding for writing netcdf
    encoding = {'fmc':{'zlib': True, "complevel": 1, 'shuffle': True}}
    
    # Loop for each function
    for func_name,func in funcs.items():
        print(f'Calculating {func_name}')
        # Location for output
        out_pth = pth+f'layers/tiles/{int(x_left)}_{int(y_top)}/fmc_nsw_{int(x_left)}_{int(y_top)}_{func_name}.nc'
        #print(out_pth)
        
        # Run function, saving output to netcdf
        if 'season' in ds.dims:
            out = func(ds,x_left,y_top)
            out = out.astype(np.float32)
            out = out.rename({'fmc':f'fmc_{func_name}'})
            out = out.assign_coords({'spatial_ref':ds.spatial_ref})
            for season in ds.season:
                out.sel(season=season).to_netcdf(out_pth[:-3]+f'_{season}.nc',encoding=encoding)
        else:
            out = func(ds,x_left,y_top)
            out = out.astype(np.float32)
            out = out.rename({'fmc':f'fmc_{func_name}'})
            out = out.assign_coords({'spatial_ref':ds.spatial_ref})
            out.to_netcdf(out_pth,encoding=encoding)
        print(f'Calculated {func_name}.')


# In[140]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Model FMC of sections of tree cover raster""")
    parser.add_argument('-x', '--x_left', type=float, required=True, help="Specifies the left x coordinate for a section")
    parser.add_argument('-y', '--y_top', type=float, required=True, help="Specifies the top y coordinate for a section")
    args = parser.parse_args()
    
    print(args.x_left, args.y_top)
    timeseries_functions(args.x_left, args.y_top)

