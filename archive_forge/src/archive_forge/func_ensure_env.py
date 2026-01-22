from contextlib import ExitStack
from functools import wraps, total_ordering
from inspect import getfullargspec as getargspec
import logging
import os
import re
import threading
import warnings
import attr
from rasterio._env import (
from rasterio._version import gdal_version
from rasterio.errors import EnvError, GDALVersionError, RasterioDeprecationWarning
from rasterio.session import Session, DummySession
def ensure_env(f):
    """A decorator that ensures an env exists before a function
    calls any GDAL C functions."""

    @wraps(f)
    def wrapper(*args, **kwds):
        if local._env:
            return f(*args, **kwds)
        else:
            with Env.from_defaults():
                return f(*args, **kwds)
    return wrapper