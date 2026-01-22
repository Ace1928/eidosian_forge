import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def from_like_context(ctx, param, value):
    """Return the value for an option from the context if the option
    or `--all` is given, else return None."""
    if ctx.obj and ctx.obj.get('like') and (value == 'like' or ctx.obj.get('all_like')):
        return ctx.obj['like'][param.name]
    else:
        return None