from collections import namedtuple
import glob
import logging
from logging import NullHandler
import os
import platform
import sys
import warnings
from rasterio._show_versions import show_versions
from rasterio._version import gdal_version, get_geos_version, get_proj_version
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension, is_blacklisted
from rasterio.dtypes import (
from rasterio.env import ensure_env_with_credentials, Env
from rasterio.errors import (
from rasterio.io import (
from rasterio.profiles import default_gtiff_profile
from rasterio.transform import Affine, guard_transform
from rasterio._path import _parse_path
import rasterio._err
import rasterio.coords
import rasterio.enums
import rasterio._path
def band(ds, bidx):
    """A dataset and one or more of its bands

    Parameters
    ----------
    ds: dataset object
        An opened rasterio dataset object.
    bidx: int or sequence of ints
        Band number(s), index starting at 1.

    Returns
    -------
    rasterio.Band
    """
    return Band(ds, bidx, set(ds.dtypes).pop(), ds.shape)