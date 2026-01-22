import sys
import numpy as np
from .dask import DaskInterface
from .interface import Interface
from .spatialpandas import SpatialPandasInterface
@classmethod
def series_type(cls):
    from spatialpandas.dask import DaskGeoSeries
    return DaskGeoSeries