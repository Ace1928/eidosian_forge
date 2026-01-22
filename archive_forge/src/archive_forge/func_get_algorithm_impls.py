import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def get_algorithm_impls(self, algorithm):
    if algorithm == 'MD5':
        H = lambda x: hashlib.md5(x.encode('ascii')).hexdigest()
    elif algorithm == 'SHA':
        H = lambda x: hashlib.sha1(x.encode('ascii')).hexdigest()
    else:
        raise ValueError('Unsupported digest authentication algorithm %r' % algorithm)
    KD = lambda s, d: H('%s:%s' % (s, d))
    return (H, KD)