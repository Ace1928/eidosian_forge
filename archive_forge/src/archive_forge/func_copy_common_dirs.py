from __future__ import absolute_import, division, print_function
import errno
import filecmp
import grp
import os
import os.path
import platform
import pwd
import shutil
import stat
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def copy_common_dirs(src, dest, module):
    changed = False
    common_dirs = filecmp.dircmp(src, dest).common_dirs
    for item in common_dirs:
        src_item_path = os.path.join(src, item)
        dest_item_path = os.path.join(dest, item)
        b_src_item_path = to_bytes(src_item_path, errors='surrogate_or_strict')
        b_dest_item_path = to_bytes(dest_item_path, errors='surrogate_or_strict')
        diff_files_changed = copy_diff_files(b_src_item_path, b_dest_item_path, module)
        left_only_changed = copy_left_only(b_src_item_path, b_dest_item_path, module)
        if diff_files_changed or left_only_changed:
            changed = True
        changed = copy_common_dirs(os.path.join(src, item), os.path.join(dest, item), module) or changed
    return changed