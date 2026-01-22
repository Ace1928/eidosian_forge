import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def edit_nodata_handler(ctx, param, value):
    """Get nodata value from a template file or command line.

    Expected values are 'like', 'null', a numeric value, 'nan', or None.

    Returns
    -------
    float or None

    Raises
    ------
    click.BadParameter

    """
    if value == 'like' or value is None:
        retval = from_like_context(ctx, param, value)
        if retval is not None:
            return retval
    return nodata_handler(ctx, param, value)