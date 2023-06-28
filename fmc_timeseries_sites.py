#!/usr/bin/env python
# coding: utf-8

# ## Based on dataframe of locations, retrieve Sentinel reflectance via DEA datacube and calculate FMC timeseries then save just result back to dataframe

# In[ ]:


# Run this in DEA Sandbox or the DEA environment on NCI


# In[4]:


import sys
sys.path.append('dea-notebooks/Scripts/') # i.e. dea-notebooks/Scripts/
import datacube
from dea_datahandling import load_ard
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import numpy as np
import time

dc = datacube.Datacube(app='sentinel_fmc')


# In[ ]:


# Set path to data in workign directory
gpath = 'data/'


# In[ ]:


# Get coordinates of locations to model FMC timeseries
gdf = gpd.read_file(gpath+'coords_select.csv')
gdf[['x','y']] = gdf[['x','y']].astype(float)


# In[ ]:


# read Look Up Table from file
lut = np.load(gpath+'lut_filtered.npy')
# the squares of the LUT
squares = np.einsum("ij,ij->i", lut[:, 1:], lut[:, 1:]) ** 0.5


# In[ ]:


# Defin reflectance bands to retrieve from datacube
measurements=['nbart_blue','nbart_green','nbart_red',
             'nbart_red_edge_1','nbart_red_edge_2','nbart_red_edge_3',
             'nbart_nir_1','nbart_nir_2','nbart_swir_2','nbart_swir_3']

# number of LUT values to consider in retrieval (optimised to 30)
top_n = 30

# Df to save results
pd.DataFrame(columns=['time', 'x', 'y', 'fmc', 'index', 'spatial_ref']).to_csv(gpath+'fmc_sites.csv')

# Define inputs for retrieval algorithym 
inputs = ['nbart_green','nbart_red','nbart_red_edge_1','nbart_red_edge_2','nbart_red_edge_3',
             'nbart_nir_1','nbart_nir_2','nbart_swir_2','nbart_swir_3','ndii']

# Loop over gdf of locations extracting timeseries datacube
for site in gdf.itertuples(): #TODO
    start = time.time()
    print(site.Index, site.y, site.x)
    
    # Query the datacube
    try:
        s2_cube = load_ard(dc=dc, products=['s2a_ard_granule','s2b_ard_granule'], crs='EPSG:3577',
                 x=(site.x-10,site.x+10), y=(site.y+10,site.y-10), time=('2015-01-01','2022-12-31'),#TODO
                   measurements=measurements, output_crs='EPSG:3577', resolution=(-20, 20),
                   mask_pixel_quality=True, group_by='solar_day')
    except ValueError:
        continue

    # Start FMC model
    s2_cube['ndii'] = ((s2_cube.nbart_nir_1-s2_cube.nbart_swir_2)/(s2_cube.nbart_nir_1+s2_cube.nbart_swir_2))
    s2_cube = s2_cube / 10000
    s2_cube['ndvi'] = ((s2_cube.nbart_nir_1-s2_cube.nbart_red)/(s2_cube.nbart_nir_1+s2_cube.nbart_red)) 
    
    # Create output array frame
    canvas = np.ones(s2_cube['ndvi'].values.shape, dtype=np.float32) * np.nan
    
    # Loop over pixels and time, retrieve FMC from LUT
    for t in range(s2_cube['ndvi'].values.shape[0]):

        for j in range(s2_cube['ndvi'].values.shape[1]):

            for i in range(s2_cube['ndvi'].values.shape[2]):
                x = s2_cube[inputs].to_array().values[:,t, j, i]
                if np.isnan(s2_cube['ndvi'][t, j, i]) or s2_cube['ndvi'][t, j, i] < 0.15:
                    continue

                θ = -1 * (
                    np.einsum("ij,j->i", lut[:, 1:], x)
                    / (np.einsum("i,i->", x, x) ** 0.5 * squares)
                )

                idxs = np.argpartition(θ, top_n)[:top_n] # find a number of closely matching spectra
                canvas[t, j, i] = np.median(lut[idxs, 0]) # takes median FMC of best matching spectra
    
    # Make new dataarray with same coordinates as input data
    s2_cube['fmc'] = (['time','y','x'], canvas)
    # Write to file by appending to fmc_sites.csv
    s2_cube[['fmc','index']].to_dataframe().reset_index().to_csv(
        gpath+'fmc_sites_filtered.csv',mode='a',header=False)
    
    print(time.time()-start)

