from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def list_connection_info(self):
    cmd = [self.nmcli_bin, '--fields', 'name', '--terse', 'con', 'show']
    rc, out, err = self.execute_command(cmd)
    if rc != 0:
        raise NmcliModuleError(err)
    return out.splitlines()