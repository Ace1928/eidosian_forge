import sys
import numpy as np
from .dask import DaskInterface
from .interface import Interface
from .spatialpandas import SpatialPandasInterface
@classmethod
def frame_type(cls):
    from spatialpandas.dask import DaskGeoDataFrame
    return DaskGeoDataFrame