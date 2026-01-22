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
def retry_http_basic_auth(self, url, realm, data=None):
    host, selector = _splithost(url)
    i = host.find('@') + 1
    host = host[i:]
    user, passwd = self.get_user_passwd(host, realm, i)
    if not (user or passwd):
        return None
    host = '%s:%s@%s' % (quote(user, safe=''), quote(passwd, safe=''), host)
    newurl = 'http://' + host + selector
    if data is None:
        return self.open(newurl)
    else:
        return self.open(newurl, data)