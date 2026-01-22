from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
def tileextent(self, quadkey):
    x_y_z = self.quadkey_to_tms(quadkey, google=True)
    return GoogleWTS.tileextent(self, x_y_z)