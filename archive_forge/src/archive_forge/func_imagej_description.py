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
def imagej_description(shape, rgb=None, colormaped=False, version='1.11a', hyperstack=None, mode=None, loop=None, **kwargs):
    """Return ImageJ image description from data shape.

    ImageJ can handle up to 6 dimensions in order TZCYXS.

    >>> imagej_description((51, 5, 2, 196, 171))  # doctest: +SKIP
    ImageJ=1.11a
    images=510
    channels=2
    slices=5
    frames=51
    hyperstack=true
    mode=grayscale
    loop=false

    """
    if colormaped:
        raise NotImplementedError('ImageJ colormapping not supported')
    shape = imagej_shape(shape, rgb=rgb)
    rgb = shape[-1] in (3, 4)
    result = ['ImageJ=%s' % version]
    append = []
    result.append('images=%i' % product(shape[:-3]))
    if hyperstack is None:
        hyperstack = True
        append.append('hyperstack=true')
    else:
        append.append('hyperstack=%s' % bool(hyperstack))
    if shape[2] > 1:
        result.append('channels=%i' % shape[2])
    if mode is None and (not rgb):
        mode = 'grayscale'
    if hyperstack and mode:
        append.append('mode=%s' % mode)
    if shape[1] > 1:
        result.append('slices=%i' % shape[1])
    if shape[0] > 1:
        result.append('frames=%i' % shape[0])
        if loop is None:
            append.append('loop=false')
    if loop is not None:
        append.append('loop=%s' % bool(loop))
    for key, value in kwargs.items():
        append.append('%s=%s' % (key.lower(), value))
    return '\n'.join(result + append + [''])