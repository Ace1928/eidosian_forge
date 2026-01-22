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
def cube_to_png(grid, vmin, vmax, file):
    tiles_data, tile_shape, rows, columns, slices = _cube_to_tiles(grid, vmin, vmax)
    image_height, image_width, __ = tiles_data.shape
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        img = PIL.Image.frombuffer('RGBA', (image_width, image_height), tiles_data, 'raw')
        img.save(file, 'png')
    return ((image_width, image_height), tile_shape, rows, columns, slices)