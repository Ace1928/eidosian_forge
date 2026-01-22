import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
def local_open(url):
    """Read a local path, with special support for directories"""
    scheme, server, path, param, query, frag = urllib.parse.urlparse(url)
    filename = urllib.request.url2pathname(path)
    if os.path.isfile(filename):
        return urllib.request.urlopen(url)
    elif path.endswith('/') and os.path.isdir(filename):
        files = []
        for f in os.listdir(filename):
            filepath = os.path.join(filename, f)
            if f == 'index.html':
                with open(filepath, 'r') as fp:
                    body = fp.read()
                break
            elif os.path.isdir(filepath):
                f += '/'
            files.append('<a href="{name}">{name}</a>'.format(name=f))
        else:
            tmpl = '<html><head><title>{url}</title></head><body>{files}</body></html>'
            body = tmpl.format(url=url, files='\n'.join(files))
        status, message = (200, 'OK')
    else:
        status, message, body = (404, 'Path not found', 'Not found')
    headers = {'content-type': 'text/html'}
    body_stream = io.StringIO(body)
    return urllib.error.HTTPError(url, status, message, headers, body_stream)