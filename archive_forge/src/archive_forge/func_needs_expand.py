from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def needs_expand(self):
    """
        Is an Expand operation required to resolve the differences between the VLAN information and the module parameters?

        The VLAN's network is expanded by reducing the size of its network prefix.

        :return: True, if an Expand operation is required; otherwise, False.
        """
    return self.private_ipv4_prefix_size_decreased