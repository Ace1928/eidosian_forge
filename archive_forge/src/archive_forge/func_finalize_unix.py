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
def finalize_unix(self):
    """Finalizes options for posix platforms."""
    if self.install_base is not None or self.install_platbase is not None:
        if self.install_lib is None and self.install_purelib is None and (self.install_platlib is None) or self.install_headers is None or self.install_scripts is None or (self.install_data is None):
            raise DistutilsOptionError('install-base or install-platbase supplied, but installation scheme is incomplete')
        return
    if self.user:
        if self.install_userbase is None:
            raise DistutilsPlatformError('User base directory is not specified')
        self.install_base = self.install_platbase = self.install_userbase
        self.select_scheme('unix_user')
    elif self.home is not None:
        self.install_base = self.install_platbase = self.home
        self.select_scheme('unix_home')
    else:
        if self.prefix is None:
            if self.exec_prefix is not None:
                raise DistutilsOptionError('must not supply exec-prefix without prefix')
            self.prefix = os.path.normpath(sys.prefix)
            self.exec_prefix = os.path.normpath(sys.exec_prefix)
        elif self.exec_prefix is None:
            self.exec_prefix = self.prefix
        self.install_base = self.prefix
        self.install_platbase = self.exec_prefix
        self.select_scheme('unix_prefix')