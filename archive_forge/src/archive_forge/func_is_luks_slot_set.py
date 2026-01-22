from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def is_luks_slot_set(self, device, keyslot):
    """ check if a keyslot is set
        """
    result = self._run_command([self._cryptsetup_bin, 'luksDump', device])
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while dumping LUKS header from %s' % (device,))
    result_luks1 = 'Key Slot %d: ENABLED' % keyslot in result[STDOUT]
    result_luks2 = ' %d: luks2' % keyslot in result[STDOUT]
    return result_luks1 or result_luks2