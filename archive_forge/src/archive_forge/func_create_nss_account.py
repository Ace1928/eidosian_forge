from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
def create_nss_account(self):
    account = dict()
    if self.parameters.get('name'):
        account['accountName'] = self.parameters['name']
    account['providerKeys'] = {'nssUserName': self.parameters['username'], 'nssPassword': self.parameters['password']}
    account['vsaList'] = []
    if self.parameters.get('vsa_list'):
        account['vsaList'] = self.parameters['vsa_list']
    response, err, second_dummy = self.rest_api.send_request('POST', '%s/accounts/nss' % self.rest_api.api_root_path, None, account, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error: unexpected response on creating nss account: %s, %s' % (str(err), str(response)))