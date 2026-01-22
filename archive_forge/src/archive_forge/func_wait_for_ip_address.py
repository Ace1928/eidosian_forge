from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.xenserver import (xenserver_common_argument_spec, XenServerObject, get_object_ref,
def wait_for_ip_address(self):
    """Waits for VM to acquire an IP address."""
    self.vm_params['guest_metrics'] = wait_for_vm_ip_address(self.module, self.vm_ref, self.module.params['state_change_timeout'])