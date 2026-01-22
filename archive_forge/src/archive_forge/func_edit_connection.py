from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def edit_connection(self):
    commands = self.edit_commands + ['save', 'quit']
    return self.execute_edit_commands(commands, arguments=[self.conn_name])