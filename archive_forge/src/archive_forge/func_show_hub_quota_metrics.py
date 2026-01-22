from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def show_hub_quota_metrics(self, resource_group, name):
    result = []
    try:
        resp = self.IoThub_client.iot_hub_resource.get_quota_metrics(resource_group, name)
        while True:
            result.append(resp.next().as_dict())
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Failed to getting quota metrics for IoT Hub {0}/{1}: {2}'.format(resource_group, name, str(exc)))
    return result