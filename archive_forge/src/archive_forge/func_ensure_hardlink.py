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
def ensure_hardlink(path, src, follow, force, timestamps):
    b_path = to_bytes(path, errors='surrogate_or_strict')
    b_src = to_bytes(src, errors='surrogate_or_strict')
    prev_state = get_state(b_path)
    file_args = module.load_file_common_arguments(module.params)
    mtime = get_timestamp_for_time(timestamps['modification_time'], timestamps['modification_time_format'])
    atime = get_timestamp_for_time(timestamps['access_time'], timestamps['access_time_format'])
    if prev_state != 'hard' and src is None:
        raise AnsibleModuleError(results={'msg': 'src is required for creating new hardlinks'})
    if src is not None and (not os.path.exists(b_src)):
        raise AnsibleModuleError(results={'msg': 'src does not exist', 'dest': path, 'src': src})
    diff = initial_diff(path, 'hard', prev_state)
    changed = False
    if prev_state == 'absent':
        changed = True
    elif prev_state == 'link':
        b_old_src = os.readlink(b_path)
        if b_old_src != b_src:
            diff['before']['src'] = to_native(b_old_src, errors='strict')
            diff['after']['src'] = src
            changed = True
    elif prev_state == 'hard':
        if src is not None and (not os.stat(b_path).st_ino == os.stat(b_src).st_ino):
            changed = True
            if not force:
                raise AnsibleModuleError(results={'msg': 'Cannot link, different hard link exists at destination', 'dest': path, 'src': src})
    elif prev_state == 'file':
        changed = True
        if not force:
            raise AnsibleModuleError(results={'msg': 'Cannot link, %s exists at destination' % prev_state, 'dest': path, 'src': src})
    elif prev_state == 'directory':
        changed = True
        if os.path.exists(b_path):
            if os.stat(b_path).st_ino == os.stat(b_src).st_ino:
                return {'path': path, 'changed': False}
            elif not force:
                raise AnsibleModuleError(results={'msg': 'Cannot link: different hard link exists at destination', 'dest': path, 'src': src})
    else:
        raise AnsibleModuleError(results={'msg': 'unexpected position reached', 'dest': path, 'src': src})
    if changed and (not module.check_mode):
        if prev_state != 'absent':
            b_tmppath = to_bytes(os.path.sep).join([os.path.dirname(b_path), to_bytes('.%s.%s.tmp' % (os.getpid(), time.time()))])
            try:
                if prev_state == 'directory':
                    if os.path.exists(b_path):
                        try:
                            os.unlink(b_path)
                        except OSError as e:
                            if e.errno != errno.ENOENT:
                                raise
                os.link(b_src, b_tmppath)
                os.rename(b_tmppath, b_path)
            except OSError as e:
                if os.path.exists(b_tmppath):
                    os.unlink(b_tmppath)
                raise AnsibleModuleError(results={'msg': 'Error while replacing: %s' % to_native(e, nonstring='simplerepr'), 'path': path})
        else:
            try:
                os.link(b_src, b_path)
            except OSError as e:
                raise AnsibleModuleError(results={'msg': 'Error while linking: %s' % to_native(e, nonstring='simplerepr'), 'path': path})
    if module.check_mode and (not os.path.exists(b_path)):
        return {'dest': path, 'src': src, 'changed': changed, 'diff': diff}
    changed = module.set_fs_attributes_if_different(file_args, changed, diff, expand=False)
    changed |= update_timestamp_for_file(file_args['path'], mtime, atime, diff)
    return {'dest': path, 'src': src, 'changed': changed, 'diff': diff}