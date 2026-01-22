from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def compute_disks_result(self, disk_instances):
    result = []
    for params, disk in disk_instances:
        disk_id = parse_resource_id(disk.get('id'))
        result.append(self.get_managed_disk(resource_group=disk_id.get('resource_group'), name=disk_id.get('resource_name')))
    return result