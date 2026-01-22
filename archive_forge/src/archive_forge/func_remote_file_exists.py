from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def remote_file_exists(self, remote_file, file_system):
    command = 'dir {0}/{1}'.format(file_system, remote_file)
    body = self._connection.run_commands(command)[0]
    if 'No such file' in body:
        return False
    else:
        return self.md5sum_check(remote_file, file_system)