from contextlib import contextmanager
import logging
import os
import math
from pathlib import Path
import warnings
import numpy as np
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.enums import Resampling
from rasterio.errors import RasterioDeprecationWarning
from rasterio import windows
from rasterio.transform import Affine
@contextmanager
def nullcontext(obj):
    try:
        yield obj
    finally:
        pass