from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def get_flash_size(self, file_system):
    command = 'dir {0}'.format(file_system)
    body = self._connection.run_commands(command)[0]
    match = re.search('(\\d+) bytes free', body)
    if match:
        bytes_free = match.group(1)
        return int(bytes_free)
    match = re.search('No such file or directory', body)
    if match:
        self._module.fail_json('Invalid nxos filesystem {0}'.format(file_system))
    else:
        self._module.fail_json('Unable to determine size of filesystem {0}'.format(file_system))