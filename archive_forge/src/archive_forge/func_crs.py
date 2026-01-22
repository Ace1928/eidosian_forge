from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
@property
def crs(self):
    """The cartopy CRS for the geometries in this feature."""
    return self._crs