from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import filter, int, map, open, str
from future.utils import as_native_str, PY2
import copy
import datetime
import re
import time
from future.backports.urllib.parse import urlparse, urlsplit, quote
from future.backports.http.client import HTTP_PORT
from calendar import timegm
def vals_sorted_by_key(adict):
    keys = sorted(adict.keys())
    return map(adict.get, keys)