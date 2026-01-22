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
def env_ctx_if_needed():
    """Return an Env if one does not exist

    Returns
    -------
    Env or a do-nothing context manager

    """
    if local._env:
        return NullContextManager()
    else:
        return Env.from_defaults()