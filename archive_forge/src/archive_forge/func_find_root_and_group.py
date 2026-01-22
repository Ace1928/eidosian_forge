from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def find_root_and_group(ds):
    """Find the root and group name of a netCDF4/h5netcdf dataset."""
    hierarchy = ()
    while ds.parent is not None:
        hierarchy = (ds.name.split('/')[-1],) + hierarchy
        ds = ds.parent
    group = '/' + '/'.join(hierarchy)
    return (ds, group)