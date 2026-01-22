from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def modify_connection(self):
    status = self.connection_update('modify')
    if status[0] == 0 and self.edit_commands:
        status = self.edit_connection()
    return status