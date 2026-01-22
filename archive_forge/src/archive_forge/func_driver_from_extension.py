import os
from rasterio._base import _raster_driver_extensions
from rasterio.env import GDALVersion, ensure_env
def driver_from_extension(path):
    """
    Attempt to auto-detect driver based on the extension.

    Parameters
    ----------
    path: str or pathlike object
        The path to the dataset to write with.

    Returns
    -------
    str:
        The name of the driver for the extension.
    """
    try:
        path = path.name
    except AttributeError:
        pass
    if GDALVersion().runtime() < GDALVersion.parse('2.0'):
        driver_extensions = {'tif': 'GTiff', 'tiff': 'GTiff', 'png': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG'}
    else:
        driver_extensions = raster_driver_extensions()
    try:
        return driver_extensions[os.path.splitext(path)[-1].lstrip('.').lower()]
    except KeyError:
        raise ValueError('Unable to detect driver. Please specify driver.')