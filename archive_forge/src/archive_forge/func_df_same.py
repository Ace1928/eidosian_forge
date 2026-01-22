import geopandas as gpd
import requests
from pathlib import Path
from zipfile import ZipFile
import tempfile
from shapely.geometry import box
def df_same(new, old, dataset, log):
    assert (new.columns == old.columns).all(), 'columns should be the same'
    if new.shape != old.shape:
        dfc = old.merge(new, on='name', how='outer', suffixes=('_old', '_new')).loc[lambda d: d.isna().any(axis=1)]
        log.append(f'### {dataset} row count changed ###\n{dfc.to_markdown()}')
        return False
    dfc = new.compare(old)
    if len(dfc) > 0:
        log.append(f'### {dataset} data changed ###\n{dfc.to_markdown()}')
    return len(dfc) == 0