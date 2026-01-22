from __future__ import annotations
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from packaging.version import Version
from xarray.core.utils import is_scalar
from xarray.namedarray.utils import is_duck_array, is_duck_dask_array
Quick wrapper to get the version of the module.