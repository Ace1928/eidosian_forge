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
def load_subset(*args):
    constraint = dict(zip(dim_names, args))
    group = self.select(**constraint)
    if np.isscalar(group):
        return group_type(([group],), group=self.group, label=self.label, vdims=self.vdims)
    data = group.reindex(kdims)
    if drop_dim and self.interface.gridded:
        data = data.columns()
    return group_type(data, **group_kwargs)