import geopandas as gpd
import requests
from pathlib import Path
from zipfile import ZipFile
import tempfile
from shapely.geometry import box
def countries_override(world_raw):
    mask = world_raw['ISO_A3'].eq('-99') & world_raw['TYPE'].isin(['Sovereign country', 'Country'])
    world_raw.loc[mask, 'ISO_A3'] = world_raw.loc[mask, 'ADM0_A3']
    return world_raw.rename(columns={'GDP_MD': 'GDP_MD_EST'})