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
def quadkey_to_tms(self, quadkey, google=False):
    assert isinstance(quadkey, str), 'quadkey must be a string'
    x = y = 0
    z = len(quadkey)
    for i in range(z, 0, -1):
        mask = 1 << i - 1
        if quadkey[z - i] == '0':
            pass
        elif quadkey[z - i] == '1':
            x |= mask
        elif quadkey[z - i] == '2':
            y |= mask
        elif quadkey[z - i] == '3':
            x |= mask
            y |= mask
        else:
            raise ValueError(f'Invalid QuadKey digit sequence: {quadkey}')
    if not google:
        y = 2 ** z - 1 - y
    return (x, y, z)