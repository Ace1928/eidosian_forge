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
def initial_diff(path, state, prev_state):
    diff = {'before': {'path': path}, 'after': {'path': path}}
    if prev_state != state:
        diff['before']['state'] = prev_state
        diff['after']['state'] = state
        if state == 'absent' and prev_state == 'directory':
            walklist = {'directories': [], 'files': []}
            b_path = to_bytes(path, errors='surrogate_or_strict')
            for base_path, sub_folders, files in os.walk(b_path):
                for folder in sub_folders:
                    folderpath = os.path.join(base_path, folder)
                    walklist['directories'].append(folderpath)
                for filename in files:
                    filepath = os.path.join(base_path, filename)
                    walklist['files'].append(filepath)
            diff['before']['path_content'] = walklist
    return diff