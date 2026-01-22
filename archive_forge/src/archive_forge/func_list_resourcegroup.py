from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_resourcegroup(self):
    result = []
    self.log('List all in {0}'.format(self.resource_group))
    try:
        response = self.network_client.private_link_services.list(self.resource_group)
        while True:
            result.append(response.next())
    except StopIteration:
        pass
    except Exception:
        pass
    return [self.service_to_dict(item) for item in result]