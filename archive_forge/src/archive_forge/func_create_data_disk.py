from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def create_data_disk(self, lun, source):
    blob_uri, disk, snapshot = self.resolve_storage_source(source)
    if blob_uri or disk or snapshot:
        snapshot_resource = self.image_models.SubResource(id=snapshot) if snapshot else None
        managed_disk = self.image_models.SubResource(id=disk) if disk else None
        return self.image_models.ImageDataDisk(lun=lun, blob_uri=blob_uri, snapshot=snapshot_resource, managed_disk=managed_disk)