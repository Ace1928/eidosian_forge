from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.pycompat import integer_types
def open_store_variable(self, var):
    data = indexing.LazilyIndexedArray(PydapArrayWrapper(var))
    return Variable(var.dimensions, data, _fix_attributes(var.attributes))