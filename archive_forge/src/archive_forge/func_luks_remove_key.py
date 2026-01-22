from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_remove_key(self):
    if self.device is None or (self._module.params['remove_keyfile'] is None and self._module.params['remove_passphrase'] is None and (self._module.params['remove_keyslot'] is None)):
        return False
    if self._module.params['state'] == 'absent':
        self._module.fail_json(msg='Contradiction in setup: Asking to remove a key from absent LUKS.')
    if self._module.params['remove_keyslot'] is not None:
        if not self._crypthandler.is_luks_slot_set(self.device, self._module.params['remove_keyslot']):
            return False
        result = self._crypthandler.luks_test_key(self.device, self._module.params['keyfile'], self._module.params['passphrase'])
        if self._crypthandler.luks_test_key(self.device, self._module.params['keyfile'], self._module.params['passphrase'], self._module.params['remove_keyslot']):
            self._module.fail_json(msg='Cannot remove keyslot with keyfile or passphrase in same slot.')
        return result
    return self._crypthandler.luks_test_key(self.device, self._module.params['remove_keyfile'], self._module.params['remove_passphrase'])