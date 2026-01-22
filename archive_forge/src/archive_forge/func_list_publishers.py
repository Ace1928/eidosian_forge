from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_publishers(self):
    response = None
    results = []
    try:
        response = self.compute_client.virtual_machine_images.list_publishers(self.location)
    except ResourceNotFoundError as exc:
        self.fail('Failed to list publishers: {0}'.format(str(exc)))
    if response:
        for item in response:
            results.append(self.serialize_obj(item, 'VirtualMachineImageResource', enum_modules=AZURE_ENUM_MODULES))
    return results