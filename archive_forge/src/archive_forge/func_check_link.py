from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def check_link(self, path):
    if os.path.lexists(path):
        if not os.path.islink(path):
            self.module.fail_json(msg='%s exists but is not a symbolic link' % path)