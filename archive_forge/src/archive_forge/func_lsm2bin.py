from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def lsm2bin(lsmfile, binfile=None, tile=(256, 256), verbose=True):
    """Convert [MP]TZCYX LSM file to series of BIN files.

    One BIN file containing 'ZCYX' data are created for each position, time,
    and tile. The position, time, and tile indices are encoded at the end
    of the filenames.

    """
    verbose = print_ if verbose else nullfunc
    if binfile is None:
        binfile = lsmfile
    elif binfile.lower() == 'none':
        binfile = None
    if binfile:
        binfile += '_(z%ic%iy%ix%i)_m%%ip%%it%%03iy%%ix%%i.bin'
    verbose('\nOpening LSM file... ', end='', flush=True)
    start_time = time.time()
    with TiffFile(lsmfile) as lsm:
        if not lsm.is_lsm:
            verbose('\n', lsm, flush=True)
            raise ValueError('not a LSM file')
        series = lsm.series[0]
        shape = series.shape
        axes = series.axes
        dtype = series.dtype
        size = product(shape) * dtype.itemsize
        verbose('%.3f s' % (time.time() - start_time))
        verbose('Image\n  axes:  %s\n  shape: %s\n  dtype: %s\n  size:  %s' % (axes, shape, dtype, format_size(size)), flush=True)
        if not series.axes.endswith('TZCYX'):
            raise ValueError('not a *TZCYX LSM file')
        verbose('Copying image from LSM to BIN files', end='', flush=True)
        start_time = time.time()
        tiles = (shape[-2] // tile[-2], shape[-1] // tile[-1])
        if binfile:
            binfile = binfile % (shape[-4], shape[-3], tile[0], tile[1])
        shape = (1,) * (7 - len(shape)) + shape
        data = numpy.empty(shape[3:], dtype=dtype)
        out = numpy.empty((shape[-4], shape[-3], tile[0], tile[1]), dtype=dtype)
        pages = iter(series.pages)
        for m in range(shape[0]):
            for p in range(shape[1]):
                for t in range(shape[2]):
                    for z in range(shape[3]):
                        data[z] = next(pages).asarray()
                    for y in range(tiles[0]):
                        for x in range(tiles[1]):
                            out[:] = data[..., y * tile[0]:(y + 1) * tile[0], x * tile[1]:(x + 1) * tile[1]]
                            if binfile:
                                out.tofile(binfile % (m, p, t, y, x))
                            verbose('.', end='', flush=True)
        verbose(' %.3f s' % (time.time() - start_time))