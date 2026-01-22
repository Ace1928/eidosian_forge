import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def check_embedded_lapack(self, info):
    """ libflame does not necessarily have a wrapper for fortran LAPACK, we need to check """
    c = customized_ccompiler()
    tmpdir = tempfile.mkdtemp()
    s = textwrap.dedent('            void zungqr_();\n            int main(int argc, const char *argv[])\n            {\n                zungqr_();\n                return 0;\n            }')
    src = os.path.join(tmpdir, 'source.c')
    out = os.path.join(tmpdir, 'a.out')
    extra_args = info.get('extra_link_args', [])
    try:
        with open(src, 'w') as f:
            f.write(s)
        obj = c.compile([src], output_dir=tmpdir)
        try:
            c.link_executable(obj, out, libraries=info['libraries'], library_dirs=info['library_dirs'], extra_postargs=extra_args)
            return True
        except distutils.ccompiler.LinkError:
            return False
    finally:
        shutil.rmtree(tmpdir)