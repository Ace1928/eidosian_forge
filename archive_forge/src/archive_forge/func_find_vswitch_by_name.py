from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
@staticmethod
def find_vswitch_by_name(host, vswitch_name):
    """
        Find and return vSwitch managed object
        Args:
            host: Host system managed object
            vswitch_name: Name of vSwitch to find

        Returns: vSwitch managed object if found, else None

        """
    for vss in host.configManager.networkSystem.networkInfo.vswitch:
        if vss.name == vswitch_name:
            return vss
    return None