import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def find_projection(match_projection):
    result = None
    for tile_matrix_set_name in matrix_set_names:
        matrix_sets = self.wmts.tilematrixsets
        tile_matrix_set = matrix_sets[tile_matrix_set_name]
        crs_urn = tile_matrix_set.crs
        tms_crs = None
        if crs_urn in _URN_TO_CRS:
            tms_crs = _URN_TO_CRS.get(crs_urn)
        elif ':EPSG:' in crs_urn:
            epsg_num = crs_urn.split(':')[-1]
            tms_crs = ccrs.epsg(int(epsg_num))
        if tms_crs == match_projection:
            result = tile_matrix_set_name
            break
    return result