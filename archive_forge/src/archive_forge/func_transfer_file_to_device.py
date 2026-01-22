from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def transfer_file_to_device(self, remote_file):
    local_file = self._module.params['local_file']
    file_system = self._module.params['file_system']
    if not self.enough_space(local_file, file_system):
        self._module.fail_json('Could not transfer file. Not enough space on device.')
    frp = remote_file
    if not file_system.startswith('bootflash:'):
        frp = '{0}{1}'.format(file_system, remote_file)
    flp = os.path.join(os.path.abspath(local_file))
    try:
        self._connection.copy_file(source=flp, destination=frp, proto='scp', timeout=self._connection.get_option('persistent_command_timeout'))
        self.result['transfer_status'] = 'Sent: File copied to remote device.'
    except Exception as exc:
        self.result['failed'] = True
        self.result['msg'] = 'Exception received : %s' % exc