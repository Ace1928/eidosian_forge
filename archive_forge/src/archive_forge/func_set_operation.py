from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def set_operation(self):
    """Sets the operation according to state"""
    if self.state == 'absent':
        self.operation = 'remove'
    elif self.state == 'present' and self.find_session_by_name() is None:
        self.operation = 'add'
    else:
        self.operation = 'edit'