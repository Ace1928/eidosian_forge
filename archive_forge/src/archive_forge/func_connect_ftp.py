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
def connect_ftp(self, user, passwd, host, port, dirs, timeout):
    key = (user, host, port, '/'.join(dirs), timeout)
    if key in self.cache:
        self.timeout[key] = time.time() + self.delay
    else:
        self.cache[key] = ftpwrapper(user, passwd, host, port, dirs, timeout)
        self.timeout[key] = time.time() + self.delay
    self.check_cache()
    return self.cache[key]