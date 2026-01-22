from __future__ import (absolute_import, division, print_function)
import os
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ctypes import CDLL, c_char_p, c_int, byref, POINTER, get_errno
def selinux_getenforcemode():
    enforcemode = c_int()
    rc = _selinux_lib.selinux_getenforcemode(byref(enforcemode))
    return [rc, enforcemode.value]