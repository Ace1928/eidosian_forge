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
def pathname2url(pathname):
    """OS-specific conversion from a file system path to a relative URL
        of the 'file' scheme; not recommended for general use."""
    return quote(pathname)