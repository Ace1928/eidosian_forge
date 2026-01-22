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
def read_tvips_header(fh, byteorder, dtype, count, offsetsize):
    """Read TVIPS EM-MENU headers and return as dict."""
    result = {}
    header = fh.read_record(TIFF.TVIPS_HEADER_V1, byteorder=byteorder)
    for name, typestr in TIFF.TVIPS_HEADER_V1:
        result[name] = header[name].tolist()
    if header['Version'] == 2:
        header = fh.read_record(TIFF.TVIPS_HEADER_V2, byteorder=byteorder)
        if header['Magic'] != int(2863311530):
            warnings.warn('invalid TVIPS v2 magic number')
            return {}
        for name, typestr in TIFF.TVIPS_HEADER_V2:
            if typestr.startswith('V'):
                s = header[name].tostring().decode('utf16', errors='ignore')
                result[name] = stripnull(s, null='\x00')
            else:
                result[name] = header[name].tolist()
        for axis in 'XY':
            header['PhysicalPixelSize' + axis] /= 1000000000.0
            header['PixelSize' + axis] /= 1000000000.0
    elif header.version != 1:
        warnings.warn('unknown TVIPS header version')
        return {}
    return result