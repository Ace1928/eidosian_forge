import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def like_handler(ctx, param, value):
    """Copy a dataset's meta property to the command context for access
    from other callbacks."""
    if ctx.obj is None:
        ctx.obj = {}
    if value:
        with rasterio.open(value) as src:
            metadata = src.meta
            ctx.obj['like'] = metadata
            ctx.obj['like']['transform'] = metadata['transform']
            ctx.obj['like']['tags'] = src.tags()
            ctx.obj['like']['colorinterp'] = src.colorinterp
        return True
    else:
        return False