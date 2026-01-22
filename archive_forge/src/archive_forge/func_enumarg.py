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
def enumarg(enum, arg):
    """Return enum member from its name or value.

    >>> enumarg(TIFF.PHOTOMETRIC, 2)
    <PHOTOMETRIC.RGB: 2>
    >>> enumarg(TIFF.PHOTOMETRIC, 'RGB')
    <PHOTOMETRIC.RGB: 2>

    """
    try:
        return enum(arg)
    except Exception:
        try:
            return enum[arg.upper()]
        except Exception:
            raise ValueError('invalid argument %s' % arg)