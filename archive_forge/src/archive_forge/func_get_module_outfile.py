import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def get_module_outfile(self, build_dir, package, module):
    outfile_path = [build_dir] + list(package) + [module + '.py']
    return os.path.join(*outfile_path)