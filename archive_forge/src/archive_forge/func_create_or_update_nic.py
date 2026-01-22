from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
from ansible.module_utils._text import to_native
def create_or_update_nic(self, nic):
    try:
        poller = self.network_client.network_interfaces.begin_create_or_update(self.resource_group, self.name, nic)
        new_nic = self.get_poller_result(poller)
        return nic_to_dict(new_nic)
    except Exception as exc:
        self.fail('Error creating or updating network interface {0} - {1}'.format(self.name, str(exc)))