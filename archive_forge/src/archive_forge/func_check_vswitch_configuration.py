from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def check_vswitch_configuration(self):
    """
        Check if vSwitch exists
        Returns: 'present' if vSwitch exists or 'absent' if not

        """
    self.vss = self.find_vswitch_by_name(self.host_system, self.switch)
    if self.vss is None:
        return 'absent'
    else:
        return 'present'