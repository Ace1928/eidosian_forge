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
def right_pad_string(source, size, pad=None):
    """right-pad or truncate <source> string, so it has length <size>"""
    cur = len(source)
    if size > cur:
        if pad is None:
            pad = _UNULL if isinstance(source, unicode) else _BNULL
        return source + pad * (size - cur)
    else:
        return source[:size]