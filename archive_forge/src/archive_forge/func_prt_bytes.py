import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
def prt_bytes(num_bytes, human_flag):
    """
    convert a number > 1024 to printable format, either in 4 char -h format as
    with ls -lh or return as 12 char right justified string
    """
    if not human_flag:
        return '%12s' % num_bytes
    num = float(num_bytes)
    suffixes = [None] + list('KMGTPEZY')
    for suffix in suffixes[:-1]:
        if num <= 1023:
            break
        num /= 1024.0
    else:
        suffix = suffixes[-1]
    if not suffix:
        return '%4s' % num_bytes
    elif num >= 10:
        return '%3d%s' % (num, suffix)
    else:
        return '%.1f%s' % (num, suffix)