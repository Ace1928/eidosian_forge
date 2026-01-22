from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def check_if_vswitch_exists(self, host_system):
    """
        Check if vSwitch exists
        Returns: 'present' if vSwitch exists or 'absent' if not
        """
    self.switch_object = self.find_vswitch_by_name(host_system=host_system, vswitch_name=self.switch)
    if self.switch_object is None:
        return 'absent'
    return 'present'