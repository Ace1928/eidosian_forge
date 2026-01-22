import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
def single_tile(self, lon, lat):
    fname = self.srtm_fname(lon, lat)
    if fname is None:
        raise ValueError('No srtm tile found for those coordinates.')
    return read_SRTM(fname)