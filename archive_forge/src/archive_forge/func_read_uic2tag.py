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
def read_uic2tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC2Tag from file and return as dict."""
    assert dtype == '2I' and byteorder == '<'
    values = fh.read_array('<u4', 6 * planecount).reshape(planecount, 6)
    return {'ZDistance': values[:, 0] / values[:, 1], 'DateCreated': values[:, 2], 'TimeCreated': values[:, 3], 'DateModified': values[:, 4], 'TimeModified': values[:, 5]}