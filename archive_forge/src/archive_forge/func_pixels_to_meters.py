from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def pixels_to_meters(self, px, py, level):
    res = self._get_resolution(level)
    mx = px * res - self.x_origin_offset
    my = py * res - self.y_origin_offset
    return (mx, my)