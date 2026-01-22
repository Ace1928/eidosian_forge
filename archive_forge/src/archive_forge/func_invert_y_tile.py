from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def invert_y_tile(y, z):
    return 2 ** z - 1 - y