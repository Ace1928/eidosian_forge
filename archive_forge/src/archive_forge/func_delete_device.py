from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_device(self, etag):
    try:
        response = self.mgmt_client.delete_device(self.name, etag=etag)
        return response
    except Exception as exc:
        self.fail('Error when deleting IoT Hub device {0}: {1}'.format(self.name, exc.message or str(exc)))