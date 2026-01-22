import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def safe_header(name, value):
    if value is not None and name in SENSITIVE_HEADERS:
        h = hashlib.sha1(value)
        d = h.hexdigest()
        return (name, '{SHA1}%s' % d)
    else:
        return (name, value)