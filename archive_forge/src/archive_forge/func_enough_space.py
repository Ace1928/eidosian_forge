from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def enough_space(self, file, file_system):
    flash_size = self.get_flash_size(file_system)
    file_size = os.path.getsize(file)
    if file_size > flash_size:
        return False
    return True