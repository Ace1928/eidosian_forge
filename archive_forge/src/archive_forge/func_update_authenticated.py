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
def update_authenticated(self, uri, is_authenticated=False):
    if isinstance(uri, str):
        uri = [uri]
    for default_port in (True, False):
        for u in uri:
            reduced_uri = self.reduce_uri(u, default_port)
            self.authenticated[reduced_uri] = is_authenticated