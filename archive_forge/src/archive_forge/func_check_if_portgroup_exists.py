from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def check_if_portgroup_exists(self, host_system):
    """
        Check if portgroup exists
        Returns: 'present' if portgroup exists or 'absent' if not
        """
    self.portgroup_object = self.find_portgroup_by_name(host_system=host_system, portgroup_name=self.portgroup, vswitch_name=self.switch)
    if self.portgroup_object is None:
        return 'absent'
    return 'present'