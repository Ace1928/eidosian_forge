from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def get_container_device_by_name(self, name):
    """ obtain device name based on the LUKS container name
            return None if not found
            raise ValueError if lsblk command fails
        """
    result = self._run_command([self._cryptsetup_bin, 'status', name])
    if result[RETURN_CODE] != 0:
        return None
    m = LUKS_DEVICE_REGEX.search(result[STDOUT])
    device = m.group(1)
    return device