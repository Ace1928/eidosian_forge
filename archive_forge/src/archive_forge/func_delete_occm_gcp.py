from __future__ import absolute_import, division, print_function
import uuid
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def delete_occm_gcp(self):
    """
        Delete Cloud Manager connector for GCP
        """
    api_url = GCP_DEPLOYMENT_MANAGER + '/deploymentmanager/v2/projects/%s/global/deployments/%s%s' % (self.parameters['project_id'], self.parameters['name'], self.gcp_common_suffix_name)
    headers = {'X-User-Token': self.rest_api.token_type + ' ' + self.rest_api.token, 'Authorization': self.rest_api.token_type + ' ' + self.rest_api.gcp_token, 'X-Tenancy-Account-Id': self.parameters['account_id'], 'Content-type': 'application/json', 'Referer': 'Ansible_NetApp'}
    response, error, dummy = self.rest_api.delete(api_url, None, header=headers)
    if error is not None:
        return 'Error: unexpected response on deleting VM: %s, %s' % (str(error), str(response))
    time.sleep(30)
    if 'client_id' not in self.parameters:
        return None
    retries = 30
    while retries > 0:
        agent, error = self.na_helper.get_occm_agent_by_id(self.rest_api, self.parameters['client_id'])
        if error is not None:
            return 'Error: Not able to get occm status after deleting VM: %s, %s' % (str(error), str(agent))
        if agent['status'] != ['active', 'pending']:
            break
        else:
            time.sleep(10)
        retries -= 1 if agent['status'] == 'active' else 5
    if retries == 0 and agent['status'] == 'active':
        return 'Taking too long for instance to finish terminating. Latest status: %s' % str(agent)
    return None