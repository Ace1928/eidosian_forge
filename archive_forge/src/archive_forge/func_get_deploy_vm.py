from __future__ import absolute_import, division, print_function
import uuid
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_deploy_vm(self):
    """
        Get Cloud Manager connector for GCP
        :return:
            Dictionary of current details if Cloud Manager connector for GCP
            None if Cloud Manager connector for GCP is not found
        """
    api_url = GCP_DEPLOYMENT_MANAGER + '/deploymentmanager/v2/projects/%s/global/deployments/%s%s' % (self.parameters['project_id'], self.parameters['name'], self.gcp_common_suffix_name)
    headers = {'X-User-Token': self.rest_api.token_type + ' ' + self.rest_api.token, 'Authorization': self.rest_api.token_type + ' ' + self.rest_api.gcp_token}
    occm_status, error, dummy = self.rest_api.get(api_url, header=headers)
    if error is not None:
        if error == '404' and b'is not found' in occm_status:
            return None
        self.module.fail_json(msg='Error: unexpected response on getting occm: %s, %s' % (str(error), str(occm_status)))
    return occm_status