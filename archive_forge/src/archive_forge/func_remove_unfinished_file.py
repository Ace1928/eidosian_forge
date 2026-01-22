from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def remove_unfinished_file(self, new_release_path):
    changed = False
    unfinished_file_path = os.path.join(new_release_path, self.unfinished_filename)
    if os.path.lexists(unfinished_file_path):
        changed = True
        if not self.module.check_mode:
            os.remove(unfinished_file_path)
    return changed