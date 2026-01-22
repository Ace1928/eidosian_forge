from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_account(self):
    account = self.get_account()
    if not account:
        self.result['changed'] = True
        if self.module.params.get('ldap_domain'):
            required_params = ['domain', 'username']
            self.module.fail_on_missing_params(required_params=required_params)
            account = self.create_ldap_account(account)
        else:
            required_params = ['email', 'username', 'password', 'first_name', 'last_name']
            self.module.fail_on_missing_params(required_params=required_params)
            account = self.create_account(account)
    return account