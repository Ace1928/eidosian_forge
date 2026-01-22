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
def cookie_property(key, serialize=lambda v: v):

    def fset(self, v):
        self[key] = serialize(v)
    return property(lambda self: self[key], fset)