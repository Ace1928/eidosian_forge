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
def read_uic3tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC3Tag from file and return as dict."""
    assert dtype == '2I' and byteorder == '<'
    values = fh.read_array('<u4', 2 * planecount).reshape(planecount, 2)
    return {'Wavelengths': values[:, 0] / values[:, 1]}