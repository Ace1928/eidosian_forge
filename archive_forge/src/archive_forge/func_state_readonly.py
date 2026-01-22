from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def state_readonly(self):
    """
        Read the target VLAN's state.
        """
    network_domain = self._get_network_domain()
    vlan = self._get_vlan(network_domain)
    if vlan:
        self.module.exit_json(vlan=vlan_to_dict(vlan), changed=False)
    else:
        self.module.fail_json(msg='VLAN "{0}" does not exist in network domain "{1}".'.format(self.name, self.network_domain_selector))