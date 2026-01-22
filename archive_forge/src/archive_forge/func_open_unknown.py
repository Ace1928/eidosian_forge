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
def open_unknown(self, fullurl, data=None):
    """Overridable interface to open unknown URL type."""
    type, url = _splittype(fullurl)
    raise OSError('url error', 'unknown url type', type)