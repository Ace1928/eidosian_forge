from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def set_network_object(self):
    """
        Set Element SW Network object
        :description: set Network object

        :return: Network object
        :rtype: object(Network object)
        """
    bond_1g_network = self.set_network_config_object(self.bond1g)
    bond_10g_network = self.set_network_config_object(self.bond10g)
    network_object = None
    if bond_1g_network is not None or bond_10g_network is not None:
        network_object = Network(bond1_g=bond_1g_network, bond10_g=bond_10g_network)
    return network_object