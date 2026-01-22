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
def not_found_in_index(self, requirement):
    if self[requirement.key]:
        meth, msg = (self.info, "Couldn't retrieve index page for %r")
    else:
        meth, msg = (self.warn, "Couldn't find index page for %r (maybe misspelled?)")
    meth(msg, requirement.unsafe_name)
    self.scan_all()