from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_create(self):
    return self.device is not None and (self._module.params['keyfile'] is not None or self._module.params['passphrase'] is not None) and (self._module.params['state'] in ('present', 'opened', 'closed')) and (not self._crypthandler.is_luks(self.device))