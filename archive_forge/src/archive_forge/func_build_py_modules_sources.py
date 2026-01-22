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
def build_py_modules_sources(self):
    if not self.py_modules:
        return
    log.info('building py_modules sources')
    new_py_modules = []
    for source in self.py_modules:
        if is_sequence(source) and len(source) == 3:
            package, module_base, source = source
            if self.inplace:
                build_dir = self.get_package_dir(package)
            else:
                build_dir = os.path.join(self.build_src, os.path.join(*package.split('.')))
            if hasattr(source, '__call__'):
                target = os.path.join(build_dir, module_base + '.py')
                source = source(target)
            if source is None:
                continue
            modules = [(package, module_base, source)]
            if package not in self.py_modules_dict:
                self.py_modules_dict[package] = []
            self.py_modules_dict[package] += modules
        else:
            new_py_modules.append(source)
    self.py_modules[:] = new_py_modules