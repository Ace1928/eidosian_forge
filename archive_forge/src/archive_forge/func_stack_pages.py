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
def stack_pages(pages, out=None, maxworkers=1, *args, **kwargs):
    """Read data from sequence of TiffPage and stack them vertically.

    Additional parameters are passed to the TiffPage.asarray function.

    """
    npages = len(pages)
    if npages == 0:
        raise ValueError('no pages')
    if npages == 1:
        return pages[0].asarray(*args, out=out, **kwargs)
    page0 = next((p for p in pages if p is not None))
    page0.asarray(validate=None)
    shape = (npages,) + page0.keyframe.shape
    dtype = page0.keyframe.dtype
    out = create_output(out, shape, dtype)
    if maxworkers is None:
        maxworkers = multiprocessing.cpu_count() // 2
    page0.parent.filehandle.lock = maxworkers > 1
    filecache = OpenFileCache(size=max(4, maxworkers), lock=page0.parent.filehandle.lock)

    def func(page, index, out=out, filecache=filecache, args=args, kwargs=kwargs):
        """Read, decode, and copy page data."""
        if page is not None:
            filecache.open(page.parent.filehandle)
            out[index] = page.asarray(*args, lock=filecache.lock, reopen=False, validate=False, **kwargs)
            filecache.close(page.parent.filehandle)
    if maxworkers < 2:
        for i, page in enumerate(pages):
            func(page, i)
    else:
        with concurrent.futures.ThreadPoolExecutor(maxworkers) as executor:
            executor.map(func, pages, range(npages))
    filecache.clear()
    page0.parent.filehandle.lock = None
    return out