from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def generate_luks_name(self, device):
    """ Generate name for luks based on device UUID ('luks-<UUID>').
            Raises ValueError when obtaining of UUID fails.
        """
    result = self._run_command([self._lsblk_bin, '-n', device, '-o', 'UUID'])
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while generating LUKS name for %s: %s' % (device, result[STDERR]))
    dev_uuid = result[STDOUT].strip()
    return 'luks-%s' % dev_uuid