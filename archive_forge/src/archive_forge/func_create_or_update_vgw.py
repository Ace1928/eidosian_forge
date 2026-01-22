from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_or_update_vgw(self, vgw):
    try:
        poller = self.network_client.virtual_network_gateways.begin_create_or_update(self.resource_group, self.name, vgw)
        new_vgw = self.get_poller_result(poller)
        return vgw_to_dict(new_vgw)
    except Exception as exc:
        self.fail('Error creating or updating virtual network gateway {0} - {1}'.format(self.name, str(exc)))