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
def recursive_set_attributes(b_path, follow, file_args, mtime, atime):
    changed = False
    try:
        for b_root, b_dirs, b_files in os.walk(b_path):
            for b_fsobj in b_dirs + b_files:
                b_fsname = os.path.join(b_root, b_fsobj)
                if not os.path.islink(b_fsname):
                    tmp_file_args = file_args.copy()
                    tmp_file_args['path'] = to_native(b_fsname, errors='surrogate_or_strict')
                    changed |= module.set_fs_attributes_if_different(tmp_file_args, changed, expand=False)
                    changed |= update_timestamp_for_file(tmp_file_args['path'], mtime, atime)
                else:
                    tmp_file_args = file_args.copy()
                    tmp_file_args['path'] = to_native(b_fsname, errors='surrogate_or_strict')
                    changed |= module.set_fs_attributes_if_different(tmp_file_args, changed, expand=False)
                    changed |= update_timestamp_for_file(tmp_file_args['path'], mtime, atime)
                    if follow:
                        b_fsname = os.path.join(b_root, os.readlink(b_fsname))
                        if os.path.exists(b_fsname):
                            if os.path.isdir(b_fsname):
                                changed |= recursive_set_attributes(b_fsname, follow, file_args, mtime, atime)
                            tmp_file_args = file_args.copy()
                            tmp_file_args['path'] = to_native(b_fsname, errors='surrogate_or_strict')
                            changed |= module.set_fs_attributes_if_different(tmp_file_args, changed, expand=False)
                            changed |= update_timestamp_for_file(tmp_file_args['path'], mtime, atime)
    except RuntimeError as e:
        raise AnsibleModuleError(results={'msg': "Could not recursively set attributes on %s. Original error was: '%s'" % (to_native(b_path), to_native(e))})
    return changed