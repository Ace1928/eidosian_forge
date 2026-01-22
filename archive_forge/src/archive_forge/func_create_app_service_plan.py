from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_app_service_plan(self):
    """
        Creates app service plan
        :return: deserialized app service plan dictionary
        """
    self.log('Create App Service Plan {0}'.format(self.plan['name']))
    try:
        sku = _normalize_sku(self.plan['sku'])
        sku_def = SkuDescription(tier=get_sku_name(sku), name=sku, capacity=self.plan.get('number_of_workers', None))
        plan_def = AppServicePlan(location=self.plan['location'], app_service_plan_name=self.plan['name'], sku=sku_def, reserved=self.plan.get('is_linux', None))
        poller = self.web_client.app_service_plans.begin_create_or_update(resource_group_name=self.plan['resource_group'], name=self.plan['name'], app_service_plan=plan_def)
        if isinstance(poller, LROPoller):
            response = self.get_poller_result(poller)
        self.log('Response : {0}'.format(response))
        return appserviceplan_to_dict(response)
    except Exception as ex:
        self.fail('Failed to create app service plan {0} in resource group {1}: {2}'.format(self.plan['name'], self.plan['resource_group'], str(ex)))