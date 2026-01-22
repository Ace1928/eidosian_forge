from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_app_service_plan(self):
    """
        Gets app service plan
        :return: deserialized app service plan dictionary
        """
    self.log('Get App Service Plan {0}'.format(self.plan['name']))
    try:
        response = self.web_client.app_service_plans.get(resource_group_name=self.plan['resource_group'], name=self.plan['name'])
        if response is not None:
            self.log('Response : {0}'.format(response))
            self.log('App Service Plan : {0} found'.format(response.name))
            return appserviceplan_to_dict(response)
    except ResourceNotFoundError:
        pass
    self.log("Didn't find app service plan {0} in resource group {1}".format(self.plan['name'], self.plan['resource_group']))
    return False