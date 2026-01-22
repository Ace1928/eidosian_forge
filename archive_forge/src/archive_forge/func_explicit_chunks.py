from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
def explicit_chunks(chunks, shape):
    """Return explicit chunks, expanding any integer member to a tuple of integers."""
    return tuple((size // chunk * (chunk,) + ((size % chunk,) if size % chunk or size == 0 else ()) if isinstance(chunk, Number) else chunk for chunk, size in zip(chunks, shape)))