from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_managed_disk(self, parameter):
    try:
        poller = self.compute_client.disks.begin_create_or_update(self.resource_group, self.name, parameter)
        aux = self.get_poller_result(poller)
        return managed_disk_to_dict(aux)
    except Exception as e:
        self.fail('Error creating the managed disk: {0}'.format(str(e)))