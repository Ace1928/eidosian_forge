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
def reshape_axes(axes, shape, newshape, unknown='Q'):
    """Return axes matching new shape.

    Unknown dimensions are labelled 'Q'.

    >>> reshape_axes('YXS', (219, 301, 1), (219, 301))
    'YX'
    >>> reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 1, 301, 1))
    'QQYQXQ'

    """
    shape = tuple(shape)
    newshape = tuple(newshape)
    if len(axes) != len(shape):
        raise ValueError('axes do not match shape')
    size = product(shape)
    newsize = product(newshape)
    if size != newsize:
        raise ValueError('cannot reshape %s to %s' % (shape, newshape))
    if not axes or not newshape:
        return ''
    lendiff = max(0, len(shape) - len(newshape))
    if lendiff:
        newshape = newshape + (1,) * lendiff
    i = len(shape) - 1
    prodns = 1
    prods = 1
    result = []
    for ns in newshape[::-1]:
        prodns *= ns
        while i > 0 and shape[i] == 1 and (ns != 1):
            i -= 1
        if ns == shape[i] and prodns == prods * shape[i]:
            prods *= shape[i]
            result.append(axes[i])
            i -= 1
        else:
            result.append(unknown)
    return ''.join(reversed(result[lendiff:]))