from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def get_device_by_label(self, label):
    """ Returns the device that holds label passed by user
        """
    self._blkid_bin = self._module.get_bin_path('blkid', True)
    label = self._module.params['label']
    if label is None:
        return None
    result = self._run_command([self._blkid_bin, '--label', label])
    if result[RETURN_CODE] != 0:
        return None
    return result[STDOUT].strip()