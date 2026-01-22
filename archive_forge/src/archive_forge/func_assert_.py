import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def assert_(expr, msg=''):
    if not expr:
        raise AssertionError(msg)