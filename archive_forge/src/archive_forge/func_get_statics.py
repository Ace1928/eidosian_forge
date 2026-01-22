from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_statics(self, resource_group, name):
    result = []
    try:
        resp = self.automation_client.statistics.list_by_automation_account(resource_group, name)
        while True:
            result.append(resp.next().as_dict())
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Error when getting statics for automation account {0}/{1}: {2}'.format(resource_group, name, exc.message))
    return result