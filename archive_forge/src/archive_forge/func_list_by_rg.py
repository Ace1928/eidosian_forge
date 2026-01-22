from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_rg(self, name):
    self.log('List resources under resource group')
    results = []
    try:
        response = self.rm_client.resources.list_by_resource_group(name)
        while True:
            results.append(response.next().as_dict())
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Error when listing resources under resource group {0}: {1}'.format(name, exc.message or str(exc)))
    return results