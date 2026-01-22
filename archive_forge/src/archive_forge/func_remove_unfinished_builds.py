from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def remove_unfinished_builds(self, releases_path):
    changes = 0
    for release in os.listdir(releases_path):
        if os.path.isfile(os.path.join(releases_path, release, self.unfinished_filename)):
            if self.module.check_mode:
                changes += 1
            else:
                changes += self.delete_path(os.path.join(releases_path, release))
    return changes