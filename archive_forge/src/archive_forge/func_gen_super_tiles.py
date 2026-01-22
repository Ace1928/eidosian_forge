from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def gen_super_tiles(extent, zoom_level, span=None):
    xmin, ymin, xmax, ymax = extent
    super_tile_size = min(2 ** 4 * 256, 2 ** zoom_level * 256)
    super_tile_def = MercatorTileDefinition(x_range=(xmin, xmax), y_range=(ymin, ymax), tile_size=super_tile_size)
    super_tiles = super_tile_def.get_tiles_by_extent(extent, zoom_level)
    for s in super_tiles:
        st_extent = s[3]
        x_range = (st_extent[0], st_extent[2])
        y_range = (st_extent[1], st_extent[3])
        yield {'level': zoom_level, 'x_range': x_range, 'y_range': y_range, 'tile_size': super_tile_def.tile_size, 'span': span}