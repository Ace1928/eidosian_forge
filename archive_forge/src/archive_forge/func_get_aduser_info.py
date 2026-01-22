from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
def get_aduser_info(self, tenant_id):
    user = {}
    self.azure_auth_graphrbac = AzureRMAuth(is_ad_resource=True)
    cred = self.azure_auth_graphrbac.azure_credentials
    base_url = self.azure_auth_graphrbac._cloud_environment.endpoints.active_directory_graph_resource_id
    client = GraphRbacManagementClient(cred, tenant_id, base_url)
    try:
        user_info = client.signed_in_user.get()
        user['name'] = user_info.user_principal_name
        user['type'] = user_info.object_type
    except GraphErrorException as e:
        self.fail('failed to get ad user info {0}'.format(str(e)))
    return user