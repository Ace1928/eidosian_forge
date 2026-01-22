from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def meters_to_pixels(self, mx, my, level):
    res = self._get_resolution(level)
    px = (mx + self.x_origin_offset) / res
    py = (my + self.y_origin_offset) / res
    return (px, py)