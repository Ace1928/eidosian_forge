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
def redirect_internal(self, url, fp, errcode, errmsg, headers, data):
    if 'location' in headers:
        newurl = headers['location']
    elif 'uri' in headers:
        newurl = headers['uri']
    else:
        return
    fp.close()
    newurl = urljoin(self.type + ':' + url, newurl)
    urlparts = urlparse(newurl)
    if urlparts.scheme not in ('http', 'https', 'ftp', ''):
        raise HTTPError(newurl, errcode, errmsg + " Redirection to url '%s' is not allowed." % newurl, headers, fp)
    return self.open(newurl)