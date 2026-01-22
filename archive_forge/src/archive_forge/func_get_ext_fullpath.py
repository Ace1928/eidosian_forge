import contextlib
import os
import re
import sys
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler, get_python_version
from distutils.sysconfig import get_config_h_filename
from distutils.dep_util import newer_group
from distutils.extension import Extension
from distutils.util import get_platform
from distutils import log
from site import USER_BASE
def get_ext_fullpath(self, ext_name):
    """Returns the path of the filename for a given extension.

        The file is located in `build_lib` or directly in the package
        (inplace option).
        """
    fullname = self.get_ext_fullname(ext_name)
    modpath = fullname.split('.')
    filename = self.get_ext_filename(modpath[-1])
    if not self.inplace:
        filename = os.path.join(*modpath[:-1] + [filename])
        return os.path.join(self.build_lib, filename)
    package = '.'.join(modpath[0:-1])
    build_py = self.get_finalized_command('build_py')
    package_dir = os.path.abspath(build_py.get_package_dir(package))
    return os.path.join(package_dir, filename)