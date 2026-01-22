from __future__ import (absolute_import, division, print_function)
import glob
import os
import pickle
import platform
import select
import shlex
import subprocess
import traceback
from ansible.module_utils.six import PY2, b
from ansible.module_utils.common.text.converters import to_bytes, to_text
def sysv_is_enabled(name, runlevel=None):
    """
    This function will check if the service name supplied
    is enabled in any of the sysv runlevels

    :arg name: name of the service to test for
    :kw runlevel: runlevel to check (default: None)
    """
    if runlevel:
        if not os.path.isdir('/etc/rc0.d/'):
            return bool(glob.glob('/etc/init.d/rc%s.d/S??%s' % (runlevel, name)))
        return bool(glob.glob('/etc/rc%s.d/S??%s' % (runlevel, name)))
    else:
        if not os.path.isdir('/etc/rc0.d/'):
            return bool(glob.glob('/etc/init.d/rc?.d/S??%s' % name))
        return bool(glob.glob('/etc/rc?.d/S??%s' % name))