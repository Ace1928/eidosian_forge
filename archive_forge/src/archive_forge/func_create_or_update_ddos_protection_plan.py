from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_ddos_protection_plan(self, params):
    """
        Create or update DDoS protection plan.
        :return: create or update DDoS protection plan instance state dictionary
        """
    self.log('create or update DDoS protection plan {0}'.format(self.name))
    try:
        poller = self.network_client.ddos_protection_plans.begin_create_or_update(resource_group_name=params.get('resource_group'), ddos_protection_plan_name=params.get('name'), parameters=params)
        result = self.get_poller_result(poller)
        self.log('Response : {0}'.format(result))
    except Exception as ex:
        self.fail('Failed to create DDoS protection plan {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
    return ddos_protection_plan_to_dict(result)