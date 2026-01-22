import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
def serialize_max_age(v):
    if isinstance(v, timedelta):
        v = str(v.seconds + v.days * 24 * 60 * 60)
    elif isinstance(v, int):
        v = str(v)
    return bytes_(v)