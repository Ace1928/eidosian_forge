from __future__ import annotations
import importlib
import warnings
from packaging.version import Version
def raise_not_implemented_error(attr_name):

    def inner_func(*args, **kwargs):
        raise NotImplementedError(f'Function {attr_name} is not implemented for dask-expr.')
    return inner_func