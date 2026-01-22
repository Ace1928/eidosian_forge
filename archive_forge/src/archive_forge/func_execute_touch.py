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
def execute_touch(path, follow, timestamps):
    b_path = to_bytes(path, errors='surrogate_or_strict')
    prev_state = get_state(b_path)
    changed = False
    result = {'dest': path}
    mtime = get_timestamp_for_time(timestamps['modification_time'], timestamps['modification_time_format'])
    atime = get_timestamp_for_time(timestamps['access_time'], timestamps['access_time_format'])
    if prev_state == 'absent':
        if module.check_mode:
            result['changed'] = True
            return result
        try:
            open(b_path, 'wb').close()
            changed = True
        except (OSError, IOError) as e:
            raise AnsibleModuleError(results={'msg': 'Error, could not touch target: %s' % to_native(e, nonstring='simplerepr'), 'path': path})
    diff = initial_diff(path, 'touch', prev_state)
    file_args = module.load_file_common_arguments(module.params)
    try:
        changed = module.set_fs_attributes_if_different(file_args, changed, diff, expand=False)
        changed |= update_timestamp_for_file(file_args['path'], mtime, atime, diff)
    except SystemExit as e:
        if e.code:
            if prev_state == 'absent':
                os.remove(b_path)
        raise
    result['changed'] = changed
    result['diff'] = diff
    return result