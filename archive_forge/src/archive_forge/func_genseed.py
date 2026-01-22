from passlib.utils.compat import JYTHON
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
from base64 import b64encode, b64decode
from codecs import lookup as _lookup_codec
from functools import update_wrapper
import itertools
import inspect
import logging; log = logging.getLogger(__name__)
import math
import os
import sys
import random
import re
import time
import timeit
import types
from warnings import warn
from passlib.utils.binary import (
from passlib.utils.decor import (
from passlib.exc import ExpectedStringError, ExpectedTypeError
from passlib.utils.compat import (add_doc, join_bytes, join_byte_values,
from passlib.exc import MissingBackendError
def genseed(value=None):
    """generate prng seed value from system resources"""
    from hashlib import sha512
    if hasattr(value, 'getstate') and hasattr(value, 'getrandbits'):
        try:
            value = value.getstate()
        except NotImplementedError:
            value = value.getrandbits(1 << 15)
    text = u('%s %s %s %.15f %.15f %s') % (value, os.getpid() if hasattr(os, 'getpid') else None, id(object()), time.time(), tick(), os.urandom(32).decode('latin-1') if has_urandom else 0)
    return int(sha512(text.encode('utf-8')).hexdigest(), 16)