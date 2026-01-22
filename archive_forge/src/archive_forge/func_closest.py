import copy
import types
from contextlib import contextmanager
from functools import wraps
import numpy as np
import pandas as pd  # noqa
import param
from param.parameterized import ParameterizedMetaclass
from .. import util as core_util
from ..accessors import Redim
from ..dimension import (
from ..element import Element
from ..ndmapping import MultiDimensionalMapping
from ..spaces import DynamicMap, HoloMap
from .array import ArrayInterface
from .cudf import cuDFInterface  # noqa (API import)
from .dask import DaskInterface  # noqa (API import)
from .dictionary import DictInterface  # noqa (API import)
from .grid import GridInterface  # noqa (API import)
from .ibis import IbisInterface  # noqa (API import)
from .image import ImageInterface  # noqa (API import)
from .interface import Interface, iloc, ndloc
from .multipath import MultiInterface  # noqa (API import)
from .pandas import PandasAPI, PandasInterface  # noqa (API import)
from .spatialpandas import SpatialPandasInterface  # noqa (API import)
from .spatialpandas_dask import DaskSpatialPandasInterface  # noqa (API import)
from .xarray import XArrayInterface  # noqa (API import)
def closest(self, coords=None, **kwargs):
    """Snaps coordinate(s) to closest coordinate in Dataset

        Args:
            coords: List of coordinates expressed as tuples
            **kwargs: Coordinates defined as keyword pairs

        Returns:
            List of tuples of the snapped coordinates

        Raises:
            NotImplementedError: Raised if snapping is not supported
        """
    if coords is None:
        coords = []
    if self.ndims > 1:
        raise NotImplementedError('Closest method currently only implemented for 1D Elements')
    if kwargs:
        if len(kwargs) > 1:
            raise NotImplementedError('Closest method currently only supports 1D indexes')
        samples = next(iter(kwargs.values()))
        coords = samples if isinstance(samples, list) else [samples]
    xs = self.dimension_values(0)
    if xs.dtype.kind in 'SO':
        raise NotImplementedError('Closest only supported for numeric types')
    idxs = [np.argmin(np.abs(xs - coord)) for coord in coords]
    return [type(s)(xs[idx]) for s, idx in zip(coords, idxs)]