from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_webapp_configuration(self, resource_group, name):
    self.log('Get web app {0} configuration'.format(name))
    response = []
    try:
        response = self.web_client.web_apps.get_configuration(resource_group_name=resource_group, name=name)
    except Exception as ex:
        request_id = ex.request_id if ex.request_id else ''
        self.fail('Error getting web app {0} configuration, request id {1} - {2}'.format(name, request_id, str(ex)))
    return response.as_dict()