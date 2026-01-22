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
@property
def creds_by_repository(self):
    sections_with_repositories = [section for section in self.sections() if self.get(section, 'repository').strip()]
    return dict(map(self._get_repo_cred, sections_with_repositories))