import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def crack2fortran(block):
    global f2py_version
    pyf = crack2fortrangen(block) + '\n'
    header = '!    -*- f90 -*-\n! Note: the context of this file is case sensitive.\n'
    footer = '\n! This file was auto-generated with f2py (version:%s).\n! See:\n! https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e\n' % f2py_version
    return header + pyf + footer