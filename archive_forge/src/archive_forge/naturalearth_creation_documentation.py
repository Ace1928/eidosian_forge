import geopandas as gpd
import requests
from pathlib import Path
from zipfile import ZipFile
import tempfile
from shapely.geometry import box

Script that generates the included dataset 'naturalearth_lowres.shp'
and 'naturalearth_cities.shp'.

Raw data: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
Current version used: see code
