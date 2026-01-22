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
@_replace_by('_tifffile.decode_packbits')
def decode_packbits(encoded):
    """Decompress PackBits encoded byte string.

    PackBits is a simple byte-oriented run-length compression scheme.

    """
    func = ord if sys.version[0] == '2' else identityfunc
    result = []
    result_extend = result.extend
    i = 0
    try:
        while True:
            n = func(encoded[i]) + 1
            i += 1
            if n < 129:
                result_extend(encoded[i:i + n])
                i += n
            elif n > 129:
                result_extend(encoded[i:i + 1] * (258 - n))
                i += 1
    except IndexError:
        pass
    return b''.join(result) if sys.version[0] == '2' else bytes(result)