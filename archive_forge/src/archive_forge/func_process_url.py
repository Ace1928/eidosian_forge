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
def process_url(self, url, retrieve=False):
    """Evaluate a URL as a possible download, and maybe retrieve it"""
    if url in self.scanned_urls and (not retrieve):
        return
    self.scanned_urls[url] = True
    if not URL_SCHEME(url):
        self.process_filename(url)
        return
    else:
        dists = list(distros_for_url(url))
        if dists:
            if not self.url_ok(url):
                return
            self.debug('Found link: %s', url)
    if dists or not retrieve or url in self.fetched_urls:
        list(map(self.add, dists))
        return
    if not self.url_ok(url):
        self.fetched_urls[url] = True
        return
    self.info('Reading %s', url)
    self.fetched_urls[url] = True
    tmpl = 'Download error on %s: %%s -- Some packages may not be found!'
    f = self.open_url(url, tmpl % url)
    if f is None:
        return
    if isinstance(f, urllib.error.HTTPError) and f.code == 401:
        self.info('Authentication error: %s' % f.msg)
    self.fetched_urls[f.url] = True
    if 'html' not in f.headers.get('content-type', '').lower():
        f.close()
        return
    base = f.url
    page = f.read()
    if not isinstance(page, str):
        if isinstance(f, urllib.error.HTTPError):
            charset = 'latin-1'
        else:
            charset = f.headers.get_param('charset') or 'latin-1'
        page = page.decode(charset, 'ignore')
    f.close()
    for match in HREF.finditer(page):
        link = urllib.parse.urljoin(base, htmldecode(match.group(1)))
        self.process_url(link)
    if url.startswith(self.index_url) and getattr(f, 'code', None) != 404:
        page = self.process_index(url, page)