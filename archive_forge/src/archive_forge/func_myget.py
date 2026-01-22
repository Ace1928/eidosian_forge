from __future__ import annotations
import pytest
import dask
from dask.context import globalmethod
def myget(dsk, keys, **kwargs):
    var[0] = var[0] + 1
    return dask.get(dsk, keys, **kwargs)