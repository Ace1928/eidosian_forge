from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def hub_query(self):
    try:
        response = None
        response = self.mgmt_client.query_iot_hub(dict(query=self.query))
        return [self.format_twin(item) for item in response.items]
    except Exception as exc:
        if hasattr(exc, 'message'):
            pass
        else:
            self.fail('Error when listing IoT Hub devices in {0}: {1}'.format(self.hub, exc))