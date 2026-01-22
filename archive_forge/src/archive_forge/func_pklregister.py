import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def pklregister(t):
    """Register a custom reducer for the type."""

    def proxy(func):
        Pickler.dispatch[t] = func
        return func
    return proxy