from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def meters_to_tile(self, mx, my, level):
    px, py = self.meters_to_pixels(mx, my, level)
    return self.pixels_to_tile(px, py, level)