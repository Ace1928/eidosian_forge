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
def tile_volume(vol, tex_size, tile_shape, vol_size):
    tex = np.zeros(tex_size, dtype=vol.dtype)
    for tileY in range(tile_shape[1]):
        for tileX in range(tile_shape[0]):
            z = tileX + tileY * tile_shape[0]
            if z >= vol_size[2]:
                break
            slice_data = vol[z]
            xoffset = tileX * vol_size[0]
            yoffset = tileY * vol_size[1]
            tex[yoffset:yoffset + vol_size[1], xoffset:xoffset + vol_size[0]] = slice_data
    return array_to_binary(tex)