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
def sysv_exists(name):
    """
    This function will return True or False depending on
    the existence of an init script corresponding to the service name supplied.

    :arg name: name of the service to test for
    """
    return os.path.exists(get_sysv_script(name))