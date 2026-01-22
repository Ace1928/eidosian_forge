from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_device(self, device):
    response = None
    try:
        if self.auth_method == 'sas':
            response = self.mgmt_client.update_device_with_sas(self.name, device['etag'], self.primary_key, self.secondary_key, self.status, iot_edge=self.edge_enabled)
        elif self.auth_method == 'self_signed':
            response = self.mgmt_client.update_device_with_certificate_authority(self.name, self.status, iot_edge=self.edge_enabled)
        elif self.auth_method == 'certificate_authority':
            response = self.mgmt_client.update_device_with_x509(self.name, device['etag'], self.primary_thumbprint, self.secondary_thumbprint, self.status, iot_edge=self.edge_enabled)
        return self.format_item(response)
    except Exception as exc:
        self.fail('Error when creating or updating IoT Hub device {0}: {1}'.format(self.name, exc.message or str(exc)))