from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import logging
import sys
from oauthlib.common import Request, urlencode, generate_nonce
from oauthlib.common import generate_timestamp, to_unicode
from . import parameters, signature
@classmethod
def register_signature_method(cls, method_name, method_callback):
    cls.SIGNATURE_METHODS[method_name] = method_callback