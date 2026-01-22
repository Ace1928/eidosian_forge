from __future__ import absolute_import
import code
import logging
import sys
import collections
import warnings
import numpy as np
import click
from . import options
import rasterio
from rasterio.plot import show, show_hist
@click.command(short_help='Open a data file and start an interpreter.')
@options.file_in_arg
@click.option('--ipython', 'interpreter', flag_value='ipython', help='Use IPython as interpreter.')
@click.option('-m', '--mode', type=click.Choice(['r', 'r+']), default='r', help="File mode (default 'r').")
@click.pass_context
def insp(ctx, input, mode, interpreter):
    """Open the input file in a Python interpreter."""
    logger = logging.getLogger()
    try:
        with ctx.obj['env']:
            with rasterio.open(input, mode) as src:
                main('Rasterio %s Interactive Inspector (Python %s)\nType "src.meta", "src.read(1)", or "help(src)" for more information.' % (rasterio.__version__, '.'.join(map(str, sys.version_info[:3]))), src, interpreter)
    except Exception:
        logger.exception('Exception caught during processing')
        raise click.Abort()