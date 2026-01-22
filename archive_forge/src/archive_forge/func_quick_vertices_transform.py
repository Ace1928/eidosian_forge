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
def quick_vertices_transform(self, vertices, src_crs):
    return_value = super().quick_vertices_transform(vertices, src_crs)
    if return_value is None and isinstance(src_crs, PlateCarree):
        self_params = self.proj4_params.copy()
        src_params = src_crs.proj4_params.copy()
        (self_params.pop('lon_0'), src_params.pop('lon_0'))
        xs, ys = (vertices[:, 0], vertices[:, 1])
        potential = self_params == src_params and self.y_limits[0] <= ys.min() and (self.y_limits[1] >= ys.max())
        if potential:
            mod = np.diff(src_crs.x_limits)[0]
            bboxes, proj_offset = self._bbox_and_offset(src_crs)
            x_lim = (xs.min(), xs.max())
            for poly in bboxes:
                for i in [-1, 0, 1, 2]:
                    offset = mod * i - proj_offset
                    if poly[0] + offset <= x_lim[0] and poly[1] + offset >= x_lim[1]:
                        return_value = vertices + [[-offset, 0]]
                        break
                if return_value is not None:
                    break
    return return_value