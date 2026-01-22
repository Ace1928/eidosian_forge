from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def needs_edit(self):
    """
        Is an Edit operation required to resolve the differences between the VLAN information and the module parameters?

        :return: True, if an Edit operation is required; otherwise, False.
        """
    return self.name_changed or self.description_changed