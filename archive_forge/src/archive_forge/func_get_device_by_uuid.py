from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def get_device_by_uuid(self, uuid):
    """ Returns the device that holds UUID passed by user
        """
    self._blkid_bin = self._module.get_bin_path('blkid', True)
    uuid = self._module.params['uuid']
    if uuid is None:
        return None
    result = self._run_command([self._blkid_bin, '--uuid', uuid])
    if result[RETURN_CODE] != 0:
        return None
    return result[STDOUT].strip()