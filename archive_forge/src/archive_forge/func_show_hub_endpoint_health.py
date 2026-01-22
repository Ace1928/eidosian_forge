from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def show_hub_endpoint_health(self, resource_group, name):
    result = []
    try:
        resp = self.IoThub_client.iot_hub_resource.get_endpoint_health(resource_group, name)
        while True:
            result.append(resp.next().as_dict())
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Failed to getting health for IoT Hub {0}/{1} routing endpoint: {2}'.format(resource_group, name, str(exc)))
    return result