import base64
import datetime
import json
import time
import warnings
import zlib
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import force_bytes
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def b64_decode(s):
    pad = b'=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)