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
def require_gdal_version(version, param=None, values=None, is_max_version=False, reason=''):
    """A decorator that ensures the called function or parameters are supported
    by the runtime version of GDAL.  Raises GDALVersionError if conditions
    are not met.

    Examples
    --------

    .. code-block:: python

        @require_gdal_version('2.2')
        def some_func():

    calling `some_func` with a runtime version of GDAL that is < 2.2 raises a
    GDALVersionErorr.

    .. code-block:: python

        @require_gdal_version('2.2', param='foo')
        def some_func(foo='bar'):

    calling `some_func` with parameter `foo` of any value on GDAL < 2.2 raises
    a GDALVersionError.

    .. code-block:: python

        @require_gdal_version('2.2', param='foo', values=('bar',))
        def some_func(foo=None):

    calling `some_func` with parameter `foo` and value `bar` on GDAL < 2.2
    raises a GDALVersionError.


    Parameters
    ------------
    version: tuple, string, or GDALVersion
    param: string (optional, default: None)
        If `values` are absent, then all use of this parameter with a value
        other than default value requires at least GDAL `version`.
    values: tuple, list, or set (optional, default: None)
        contains values that require at least GDAL `version`.  `param`
        is required for `values`.
    is_max_version: bool (optional, default: False)
        if `True` indicates that the version provided is the maximum version
        allowed, instead of requiring at least that version.
    reason: string (optional: default: '')
        custom error message presented to user in addition to message about
        GDAL version.  Use this to provide an explanation of what changed
        if necessary context to the user.

    Returns
    ---------
    wrapped function
    """
    if values is not None:
        if param is None:
            raise ValueError('require_gdal_version: param must be provided with values')
        if not isinstance(values, (tuple, list, set)):
            raise ValueError('require_gdal_version: values must be a tuple, list, or set')
    version = GDALVersion.parse(version)
    runtime = GDALVersion.runtime()
    inequality = '>=' if runtime < version else '<='
    reason = '\n{0}'.format(reason) if reason else reason

    def decorator(f):

        @wraps(f)
        def wrapper(*args, **kwds):
            if runtime < version and (not is_max_version) or (is_max_version and runtime > version):
                if param is None:
                    raise GDALVersionError('GDAL version must be {0} {1}{2}'.format(inequality, str(version), reason))
                argspec = getargspec(f)
                full_kwds = kwds.copy()
                if argspec.args:
                    full_kwds.update(dict(zip(argspec.args[:len(args)], args)))
                if argspec.defaults:
                    defaults = dict(zip(reversed(argspec.args), reversed(argspec.defaults)))
                else:
                    defaults = {}
                if param in full_kwds:
                    if values is None:
                        if param not in defaults or full_kwds[param] != defaults[param]:
                            raise GDALVersionError('usage of parameter "{0}" requires GDAL {1} {2}{3}'.format(param, inequality, version, reason))
                    elif full_kwds[param] in values:
                        raise GDALVersionError('parameter "{0}={1}" requires GDAL {2} {3}{4}'.format(param, full_kwds[param], inequality, version, reason))
            return f(*args, **kwds)
        return wrapper
    return decorator