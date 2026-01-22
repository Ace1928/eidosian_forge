from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_plan(self):
    """
        Creates app service plan
        :return: deserialized app service plan dictionary
        """
    self.log('Create App Service Plan {0}'.format(self.name))
    try:
        sku = _normalize_sku(self.sku)
        sku_def = SkuDescription(tier=get_sku_name(sku), name=sku, capacity=self.number_of_workers)
        plan_def = AppServicePlan(location=self.location, app_service_plan_name=self.name, sku=sku_def, reserved=self.is_linux, tags=self.tags if self.tags else None)
        response = self.web_client.app_service_plans.begin_create_or_update(resource_group_name=self.resource_group, name=self.name, app_service_plan=plan_def)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        self.log('Response : {0}'.format(response))
        return appserviceplan_to_dict(response)
    except Exception as ex:
        self.fail('Failed to create app service plan {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))