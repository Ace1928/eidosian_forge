from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def pixels_to_tile(self, px, py, level):
    tx = math.ceil(px / self.tile_size)
    tx = tx if tx == 0 else tx - 1
    ty = max(math.ceil(py / self.tile_size) - 1, 0)
    return (int(tx), invert_y_tile(int(ty), level))