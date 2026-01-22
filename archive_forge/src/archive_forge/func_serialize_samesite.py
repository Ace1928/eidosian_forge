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
def serialize_samesite(v):
    v = bytes_(v)
    if SAMESITE_VALIDATION:
        if v.lower() not in (b'strict', b'lax', b'none'):
            raise ValueError("SameSite must be 'strict', 'lax', or 'none'")
    return v