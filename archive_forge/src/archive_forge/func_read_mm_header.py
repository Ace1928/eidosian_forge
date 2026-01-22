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
def read_mm_header(fh, byteorder, dtype, count, offsetsize):
    """Read FluoView mm_header tag from file and return as dict."""
    mmh = fh.read_record(TIFF.MM_HEADER, byteorder=byteorder)
    mmh = recarray2dict(mmh)
    mmh['Dimensions'] = [(bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip()) for d in mmh['Dimensions']]
    d = mmh['GrayChannel']
    mmh['GrayChannel'] = (bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip())
    return mmh