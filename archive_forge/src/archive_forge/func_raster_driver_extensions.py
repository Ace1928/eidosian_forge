import os
from rasterio._base import _raster_driver_extensions
from rasterio.env import GDALVersion, ensure_env
@ensure_env
def raster_driver_extensions():
    """
    Returns
    -------
    dict:
        Map of extensions to the driver.
    """
    return _raster_driver_extensions()