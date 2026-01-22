from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def discover_aws_fsx(self):
    """
        discover aws_fsx
        """
    api = '/fsx-ontap/working-environments/%s/discover?credentials-id=%s&workspace-id=%s&region=%s' % (self.parameters['tenant_id'], self.aws_credentials_id, self.parameters['workspace_id'], self.parameters['region'])
    response, error, dummy = self.rest_api.get(api, None, header=self.headers)
    if error:
        return 'Error: discovering aws_fsx %s' % error
    id_found = False
    for each in response:
        if each['id'] == self.parameters['file_system_id']:
            id_found = True
            break
    if not id_found:
        return 'Error: file_system_id provided could not be found'