from __future__ import absolute_import, division, print_function
import errno
import os
import shutil
import sys
import time
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def update_timestamp_for_file(path, mtime, atime, diff=None):
    b_path = to_bytes(path, errors='surrogate_or_strict')
    try:
        if mtime is Sentinel and atime is Sentinel:
            mtime = atime = time.time()
            previous_mtime = os.stat(b_path).st_mtime
            previous_atime = os.stat(b_path).st_atime
            set_time = None
        else:
            if mtime is None and atime is None:
                return False
            previous_mtime = os.stat(b_path).st_mtime
            previous_atime = os.stat(b_path).st_atime
            if mtime is None:
                mtime = previous_mtime
            elif mtime is Sentinel:
                mtime = time.time()
            if atime is None:
                atime = previous_atime
            elif atime is Sentinel:
                atime = time.time()
            if mtime == previous_mtime and atime == previous_atime:
                return False
            set_time = (atime, mtime)
        if not module.check_mode:
            os.utime(b_path, set_time)
        if diff is not None:
            if 'before' not in diff:
                diff['before'] = {}
            if 'after' not in diff:
                diff['after'] = {}
            if mtime != previous_mtime:
                diff['before']['mtime'] = previous_mtime
                diff['after']['mtime'] = mtime
            if atime != previous_atime:
                diff['before']['atime'] = previous_atime
                diff['after']['atime'] = atime
    except OSError as e:
        raise AnsibleModuleError(results={'msg': 'Error while updating modification or access time: %s' % to_native(e, nonstring='simplerepr'), 'path': path})
    return True