from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import (
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
def load_verify_locations(self, cafile=None, capath=None, cadata=None):
    self.ca_certs = cafile
    if capath is not None:
        raise SSLError('CA directories not supported in older Pythons')
    if cadata is not None:
        raise SSLError('CA data not supported in older Pythons')