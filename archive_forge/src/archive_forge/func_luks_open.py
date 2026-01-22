from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_open(self):
    if self._module.params['keyfile'] is None and self._module.params['passphrase'] is None or self.device is None or self._module.params['state'] != 'opened':
        return False
    name = self.opened_luks_name()
    if name is None:
        return True
    return False