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
def find_images(self, target_domain, target_z, start_tile=None):
    """
        Find all the quadtrees at the given target zoom, in the given
        target domain.

        target_z must be a value >= 1.

        """
    if target_z == 0:
        raise ValueError('The empty quadtree cannot be returned.')
    if start_tile is None:
        start_tiles = ['0', '1', '2', '3']
    else:
        start_tiles = [start_tile]
    for start_tile in start_tiles:
        start_tile = self.quadkey_to_tms(start_tile, google=True)
        for tile in GoogleWTS.find_images(self, target_domain, target_z, start_tile=start_tile):
            yield self.tms_to_quadkey(tile, google=True)