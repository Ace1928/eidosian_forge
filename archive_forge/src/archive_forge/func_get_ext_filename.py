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
def get_ext_filename(self, ext_name):
    """Convert the name of an extension (eg. "foo.bar") into the name
        of the file from which it will be loaded (eg. "foo/bar.so", or
        "foo\\bar.pyd").
        """
    from distutils.sysconfig import get_config_var
    ext_path = ext_name.split('.')
    ext_suffix = get_config_var('EXT_SUFFIX')
    return os.path.join(*ext_path) + ext_suffix