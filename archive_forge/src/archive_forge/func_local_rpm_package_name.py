from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import (
from ansible.module_utils.common.text.converters import to_native
def local_rpm_package_name(path):
    """return package name of a local rpm passed in.
    Inspired by ansible.builtin.yum"""
    ts = rpm.TransactionSet()
    ts.setVSFlags(rpm._RPMVSF_NOSIGNATURES)
    fd = os.open(path, os.O_RDONLY)
    try:
        header = ts.hdrFromFdno(fd)
    except rpm.error as e:
        return None
    finally:
        os.close(fd)
    return to_native(header[rpm.RPMTAG_NAME])