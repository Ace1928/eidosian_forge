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
def ensure_file_attributes(path, follow, timestamps):
    b_path = to_bytes(path, errors='surrogate_or_strict')
    prev_state = get_state(b_path)
    file_args = module.load_file_common_arguments(module.params)
    mtime = get_timestamp_for_time(timestamps['modification_time'], timestamps['modification_time_format'])
    atime = get_timestamp_for_time(timestamps['access_time'], timestamps['access_time_format'])
    if prev_state != 'file':
        if follow and prev_state == 'link':
            b_path = os.path.realpath(b_path)
            path = to_native(b_path, errors='strict')
            prev_state = get_state(b_path)
            file_args['path'] = path
    if prev_state not in ('file', 'hard'):
        raise AnsibleModuleError(results={'msg': 'file (%s) is %s, cannot continue' % (path, prev_state), 'path': path, 'state': prev_state})
    diff = initial_diff(path, 'file', prev_state)
    changed = module.set_fs_attributes_if_different(file_args, False, diff, expand=False)
    changed |= update_timestamp_for_file(file_args['path'], mtime, atime, diff)
    return {'path': path, 'changed': changed, 'diff': diff}