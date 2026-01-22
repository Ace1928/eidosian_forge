from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def makes_valid_ring(line_string):
    if len(line_string.coords) == 3:
        coords = list(line_string.coords)
        return coords[0] != coords[-1] and line_string.is_valid
    else:
        return len(line_string.coords) > 3 and line_string.is_valid