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
def open_data(self, url, data=None):
    """Use "data" URL."""
    if not isinstance(url, str):
        raise URLError('data error: proxy support for data protocol currently not implemented')
    try:
        [type, data] = url.split(',', 1)
    except ValueError:
        raise OSError('data error', 'bad data URL')
    if not type:
        type = 'text/plain;charset=US-ASCII'
    semi = type.rfind(';')
    if semi >= 0 and '=' not in type[semi:]:
        encoding = type[semi + 1:]
        type = type[:semi]
    else:
        encoding = ''
    msg = []
    msg.append('Date: %s' % time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(time.time())))
    msg.append('Content-type: %s' % type)
    if encoding == 'base64':
        data = base64.decodebytes(data.encode('ascii')).decode('latin-1')
    else:
        data = unquote(data)
    msg.append('Content-Length: %d' % len(data))
    msg.append('')
    msg.append(data)
    msg = '\n'.join(msg)
    headers = email.message_from_string(msg)
    f = io.StringIO(msg)
    return addinfourl(f, headers, url)