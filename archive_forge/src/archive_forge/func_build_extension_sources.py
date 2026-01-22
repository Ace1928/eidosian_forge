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
def build_extension_sources(self, ext):
    sources = list(ext.sources)
    log.info('building extension "%s" sources' % ext.name)
    fullname = self.get_ext_fullname(ext.name)
    modpath = fullname.split('.')
    package = '.'.join(modpath[0:-1])
    if self.inplace:
        self.ext_target_dir = self.get_package_dir(package)
    sources = self.generate_sources(sources, ext)
    sources = self.template_sources(sources, ext)
    sources = self.swig_sources(sources, ext)
    sources = self.f2py_sources(sources, ext)
    sources = self.pyrex_sources(sources, ext)
    sources, py_files = self.filter_py_files(sources)
    if package not in self.py_modules_dict:
        self.py_modules_dict[package] = []
    modules = []
    for f in py_files:
        module = os.path.splitext(os.path.basename(f))[0]
        modules.append((package, module, f))
    self.py_modules_dict[package] += modules
    sources, h_files = self.filter_h_files(sources)
    if h_files:
        log.info('%s - nothing done with h_files = %s', package, h_files)
    ext.sources = sources