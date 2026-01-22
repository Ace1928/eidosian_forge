import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def subst_vars(target, source, d):
    """Substitute any occurrence of @foo@ by d['foo'] from source file into
    target."""
    var = re.compile('@([a-zA-Z_]+)@')
    with open(source, 'r') as fs:
        with open(target, 'w') as ft:
            for l in fs:
                m = var.search(l)
                if m:
                    ft.write(l.replace('@%s@' % m.group(1), d[m.group(1)]))
                else:
                    ft.write(l)