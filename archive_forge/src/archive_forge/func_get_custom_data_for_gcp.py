from __future__ import absolute_import, division, print_function
import uuid
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_custom_data_for_gcp(self, proxy_certificates):
    """
        get custom data for GCP
        """
    if 'account_id' not in self.parameters:
        response, error = self.na_helper.get_or_create_account(self.rest_api)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on getting account: %s, %s' % (str(error), str(response)))
        self.parameters['account_id'] = response
    response, error = self.na_helper.register_agent_to_service(self.rest_api, 'GCP', '')
    if error is not None:
        self.module.fail_json(msg='Error: register agent to service for gcp failed: %s, %s' % (str(error), str(response)))
    client_id = response['clientId']
    client_secret = response['clientSecret']
    u_data = {'instanceName': self.parameters['name'], 'company': self.parameters['company'], 'clientId': client_id, 'clientSecret': client_secret, 'systemId': UUID, 'tenancyAccountId': self.parameters['account_id'], 'proxySettings': {'proxyPassword': self.parameters.get('proxy_password'), 'proxyUserName': self.parameters.get('proxy_user_name'), 'proxyUrl': self.parameters.get('proxy_url'), 'proxyCertificates': proxy_certificates}}
    user_data = json.dumps(u_data)
    return (user_data, client_id, None)