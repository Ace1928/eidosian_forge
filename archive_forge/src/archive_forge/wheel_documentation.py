import email
import itertools
import functools
import os
import posixpath
import re
import zipfile
import contextlib
from distutils.util import get_platform
import setuptools
from setuptools.extern.packaging.version import Version as parse_version
from setuptools.extern.packaging.tags import sys_tags
from setuptools.extern.packaging.utils import canonicalize_name
from setuptools.command.egg_info import write_requirements, _egg_basename
from setuptools.archive_util import _unpack_zipfile_obj
Move data entries to their correct location.