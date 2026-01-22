import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
def publish_graphics(self, graph_dir, isolate_svgs=True):
    """Wrap graphic file data for presentation in IPython.

        This method is deprecated. Use `display_figures` or
        'display_figures_svg` instead.

        graph_dir : str
            Probably provided by some tmpdir call
        isolate_svgs : bool
            Enable SVG namespace isolation in metadata"""
    warnings.warn('Use method fetch_figures.', DeprecationWarning)
    images = []
    display_data = []
    md = {}
    if self.device == 'png':
        for imgfile in sorted(glob('%s/Rplots*png' % graph_dir)):
            if os.stat(imgfile).st_size >= 1000:
                with open(imgfile, 'rb') as fh_img:
                    images.append(fh_img.read())
    else:
        imgfile = '%s/Rplot.svg' % graph_dir
        if os.stat(imgfile).st_size >= 1000:
            with open(imgfile, 'rb') as fh_img:
                images.append(fh_img.read().decode())
    mimetypes = {'png': 'image/png', 'svg': 'image/svg+xml'}
    mime = mimetypes[self.device]
    if images and self.device == 'svg' and isolate_svgs:
        md = {'image/svg+xml': dict(isolated=True)}
    for image in images:
        sys.stdout.flush()
        sys.stderr.flush()
        display_data.append(('RMagic.R', {mime: image}))
    return (display_data, md)