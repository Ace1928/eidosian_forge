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
def imsave(file, data=None, shape=None, dtype=None, bigsize=2 ** 32 - 2 ** 25, **kwargs):
    """Write numpy array to TIFF file.

    Refer to the TiffWriter class and member functions for documentation.

    Parameters
    ----------
    file : str or binary stream
        File name or writable binary stream, such as an open file or BytesIO.
    data : array_like
        Input image. The last dimensions are assumed to be image depth,
        height, width, and samples.
        If None, an empty array of the specified shape and dtype is
        saved to file.
        Unless 'byteorder' is specified in 'kwargs', the TIFF file byte order
        is determined from the data's dtype or the dtype argument.
    shape : tuple
        If 'data' is None, shape of an empty array to save to the file.
    dtype : numpy.dtype
        If 'data' is None, data-type of an empty array to save to the file.
    bigsize : int
        Create a BigTIFF file if the size of data in bytes is larger than
        this threshold and 'imagej' or 'truncate' are not enabled.
        By default, the threshold is 4 GB minus 32 MB reserved for metadata.
        Use the 'bigtiff' parameter to explicitly specify the type of
        file created.
    kwargs : dict
        Parameters 'append', 'byteorder', 'bigtiff', and 'imagej', are passed
        to TiffWriter(). Other parameters are passed to TiffWriter.save().

    Returns
    -------
    If the image data are written contiguously, return offset and bytecount
    of image data in the file.

    Examples
    --------
    >>> # save a RGB image
    >>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
    >>> imsave('temp.tif', data, photometric='rgb')

    >>> # save a random array and metadata, using compression
    >>> data = numpy.random.rand(2, 5, 3, 301, 219)
    >>> imsave('temp.tif', data, compress=6, metadata={'axes': 'TZCYX'})

    """
    tifargs = parse_kwargs(kwargs, 'append', 'bigtiff', 'byteorder', 'imagej')
    if data is None:
        size = product(shape) * numpy.dtype(dtype).itemsize
        byteorder = numpy.dtype(dtype).byteorder
    else:
        try:
            size = data.nbytes
            byteorder = data.dtype.byteorder
        except Exception:
            size = 0
            byteorder = None
    if size > bigsize and 'bigtiff' not in tifargs and (not (tifargs.get('imagej', False) or tifargs.get('truncate', False))):
        tifargs['bigtiff'] = True
    if 'byteorder' not in tifargs:
        tifargs['byteorder'] = byteorder
    with TiffWriter(file, **tifargs) as tif:
        return tif.save(data, shape, dtype, **kwargs)