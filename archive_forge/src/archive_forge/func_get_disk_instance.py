from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_disk_instance(self, managed_disk):
    resource_group = self.get_resource_group(managed_disk.get('resource_group'))
    managed_disk['location'] = managed_disk.get('location') or resource_group.location
    disk_instance = self.get_managed_disk(resource_group=managed_disk.get('resource_group'), name=managed_disk.get('name'))
    if disk_instance is not None:
        for key in ('create_option', 'source_uri', 'disk_size_gb', 'os_type', 'zone'):
            if managed_disk.get(key) is None:
                managed_disk[key] = disk_instance.get(key)
    parameter = self.generate_disk_parameters(tags=self.tags, **managed_disk)
    return (parameter, disk_instance)