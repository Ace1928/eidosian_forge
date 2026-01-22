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
@lazyattr
def mdgel_metadata(self):
    """Return consolidated metadata from MD GEL tags as dict."""
    for page in self.pages[:2]:
        if 'MDFileTag' in page.tags:
            tags = page.tags
            break
    else:
        return
    result = {}
    for code in range(33445, 33453):
        name = TIFF.TAGS[code]
        if name not in tags:
            continue
        result[name[2:]] = tags[name].value
    return result