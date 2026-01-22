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
@unique_values
def find_external_links(url, page):
    """Find rel="homepage" and rel="download" links in `page`, yielding URLs"""
    for match in REL.finditer(page):
        tag, rel = match.groups()
        rels = set(map(str.strip, rel.lower().split(',')))
        if 'homepage' in rels or 'download' in rels:
            for match in HREF.finditer(tag):
                yield urllib.parse.urljoin(url, htmldecode(match.group(1)))
    for tag in ('<th>Home Page', '<th>Download URL'):
        pos = page.find(tag)
        if pos != -1:
            match = HREF.search(page, pos)
            if match:
                yield urllib.parse.urljoin(url, htmldecode(match.group(1)))