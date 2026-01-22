from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def enable_account(self):
    account = self.get_account()
    if not account:
        account = self.present_account()
    if account['state'].lower() != 'enabled':
        self.result['changed'] = True
        args = {'id': account['id'], 'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id')}
        if not self.module.check_mode:
            res = self.query_api('enableAccount', **args)
            account = res['account']
    return account