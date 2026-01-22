from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def get_port_group_by_name(self, host_system, portgroup_name, vswitch_name):
    """
        Get specific port group by given name
        Args:
            host_system: Name of Host System
            portgroup_name: Name of Port Group
            vswitch_name: Name of the vSwitch

        Returns: List of port groups by given specifications

        """
    portgroups = self.get_all_port_groups_by_host(host_system=host_system)
    for portgroup in portgroups:
        if portgroup.spec.vswitchName == vswitch_name and portgroup.spec.name == portgroup_name:
            return portgroup
    return None