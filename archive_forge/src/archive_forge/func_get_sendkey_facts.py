from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def get_sendkey_facts(self, vm_obj, returned_value=0):
    sendkey_facts = dict()
    if vm_obj is not None:
        sendkey_facts = dict(virtual_machine=vm_obj.name, keys_send=self.params['keys_send'], string_send=self.params['string_send'], keys_send_number=self.num_keys_send, returned_keys_send_number=returned_value)
    return sendkey_facts