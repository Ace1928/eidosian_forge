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
def to_proj4_params(self):
    """
        Create an OrderedDict of key value pairs which represents this globe
        in terms of proj params.

        """
    proj4_params = (['datum', self.datum], ['ellps', self.ellipse], ['a', self.semimajor_axis], ['b', self.semiminor_axis], ['f', self.flattening], ['rf', self.inverse_flattening], ['towgs84', self.towgs84], ['nadgrids', self.nadgrids])
    return OrderedDict(((k, v) for k, v in proj4_params if v is not None))