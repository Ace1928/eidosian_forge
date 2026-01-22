from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def cube_to_tiles(grid, obj=None):
    if grid is None or len(grid.shape) == 1:
        return None
    tiles_data, slice_shape, rows, columns, slices = _cube_to_tiles(grid, obj.data_min, obj.data_max)
    image_height, image_width, __ = tiles_data.shape
    image_shape = (image_width, image_height)
    json = {'tiles': memoryview(tiles_data), 'image_shape': image_shape, 'slice_shape': slice_shape, 'rows': rows, 'columns': columns, 'slices': slices}
    return json