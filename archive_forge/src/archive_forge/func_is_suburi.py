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
def is_suburi(self, base, test):
    """Check if test is below base in a URI tree

        Both args must be URIs in reduced form.
        """
    if base == test:
        return True
    if base[0] != test[0]:
        return False
    prefix = base[1]
    if prefix[-1:] != '/':
        prefix += '/'
    return test[1].startswith(prefix)