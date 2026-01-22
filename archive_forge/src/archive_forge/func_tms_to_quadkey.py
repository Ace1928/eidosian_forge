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
def tms_to_quadkey(self, tms, google=False):
    quadKey = ''
    x, y, z = tms
    if not google:
        y = 2 ** z - 1 - y
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << i - 1
        if x & mask != 0:
            digit += 1
        if y & mask != 0:
            digit += 2
        quadKey += str(digit)
    return quadKey