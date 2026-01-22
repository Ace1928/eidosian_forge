from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def delete_aws_fsx(self, id, tenant_id):
    """
        Delete AWS FSx
        """
    api_url = '/fsx-ontap/working-environments/%s/%s' % (tenant_id, id)
    response, error, dummy = self.rest_api.delete(api_url, None, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on deleting AWS FSx: %s, %s' % (str(error), str(response)))