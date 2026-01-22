from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
def namespace_directories(self, depth=None):
    """return list of directories where the namespace should be created /
        can be found
        """
    res = []
    for index, d in enumerate(self.split[:depth]):
        if index > 0:
            d = os.path.join(*d.split('.'))
        res.append('.' + d)
    return res