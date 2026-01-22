import os
import re
import time
import hashlib
import threading
import warnings
from base64 import b64encode
from .compat import urlparse, str, basestring
from .cookies import extract_cookies_to_jar
from ._internal_utils import to_native_string
from .utils import parse_dict_header
def sha512_utf8(x):
    if isinstance(x, str):
        x = x.encode('utf-8')
    return hashlib.sha512(x).hexdigest()