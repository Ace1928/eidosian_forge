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
def process_filename(self, fn, nested=False):
    if not os.path.exists(fn):
        self.warn('Not found: %s', fn)
        return
    if os.path.isdir(fn) and (not nested):
        path = os.path.realpath(fn)
        for item in os.listdir(path):
            self.process_filename(os.path.join(path, item), True)
    dists = distros_for_filename(fn)
    if dists:
        self.debug('Found: %s', fn)
        list(map(self.add, dists))