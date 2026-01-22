from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_twin(self):
    try:
        response = self.mgmt_client.get_twin(self.name)
        return self.format_twin(response)
    except Exception as exc:
        self.fail('Error when getting IoT Hub device {0} twin: {1}'.format(self.name, exc.message or str(exc)))