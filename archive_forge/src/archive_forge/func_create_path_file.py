import sys
import sysconfig
import os
import re
from distutils import log
from distutils.core import Command
from distutils.debug import DEBUG
from distutils.sysconfig import get_config_vars
from distutils.errors import DistutilsPlatformError
from distutils.file_util import write_file
from distutils.util import convert_path, subst_vars, change_root
from distutils.util import get_platform
from distutils.errors import DistutilsOptionError
from site import USER_BASE
from site import USER_SITE
def create_path_file(self):
    """Creates the .pth file"""
    filename = os.path.join(self.install_libbase, self.path_file + '.pth')
    if self.install_path_file:
        self.execute(write_file, (filename, [self.extra_dirs]), 'creating %s' % filename)
    else:
        self.warn("path file '%s' not created" % filename)