from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_virtual_machines(self, config):
    pollers = []
    for resource_group, name, params in config:
        try:
            poller = self.compute_client.virtual_machines.begin_create_or_update(resource_group, name, params)
            pollers.append(poller)
        except AzureError as exc:
            self.fail('Error updating virtual machine (attaching/detaching disks) {0}/{1} - {2}'.format(resource_group, name, exc.message))
    return self.get_multiple_pollers_results(pollers)