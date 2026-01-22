from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def down_connection(self):
    cmd = [self.nmcli_bin, 'con', 'down', self.conn_name]
    return self.execute_command(cmd)