from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
def scale_from_extent(self, extent):
    scale = self._default_scale
    if extent is not None:
        width = abs(extent[1] - extent[0])
        height = abs(extent[3] - extent[2])
        min_extent = min(width, height)
        if min_extent != 0:
            for scale_candidate, upper_bound in self._limits:
                if min_extent <= upper_bound:
                    scale = scale_candidate
                else:
                    break
    self._scale = scale
    return self._scale