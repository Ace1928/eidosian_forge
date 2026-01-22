from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def is_valid_tile(self, x, y, z):
    if x < 0 or x >= math.pow(2, z):
        return False
    if y < 0 or y >= math.pow(2, z):
        return False
    return True