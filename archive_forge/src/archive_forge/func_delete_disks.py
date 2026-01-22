from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_disks(self, ids):
    pollers = []
    for disk_id in ids:
        try:
            disk = parse_resource_id(disk_id)
            resource_group, name = (disk.get('resource_group'), disk.get('resource_name'))
            poller = self.compute_client.disks.begin_delete(resource_group, name)
            pollers.append(poller)
        except Exception as e:
            self.fail('Error deleting the managed disk {0}/{1}: {2}'.format(resource_group, name, str(e)))
    return self.get_multiple_pollers_results(pollers)