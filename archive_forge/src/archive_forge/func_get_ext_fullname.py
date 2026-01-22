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
def get_ext_fullname(self, ext_name):
    """Returns the fullname of a given extension name.

        Adds the `package.` prefix"""
    if self.package is None:
        return ext_name
    else:
        return self.package + '.' + ext_name