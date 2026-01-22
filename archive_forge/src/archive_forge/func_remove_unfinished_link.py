from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def remove_unfinished_link(self, path):
    changed = False
    if not self.release:
        return changed
    tmp_link_name = os.path.join(path, self.release + '.' + self.unfinished_filename)
    if not self.module.check_mode and os.path.exists(tmp_link_name):
        changed = True
        os.remove(tmp_link_name)
    return changed