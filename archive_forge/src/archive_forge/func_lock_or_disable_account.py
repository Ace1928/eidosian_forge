from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def lock_or_disable_account(self, lock=False):
    account = self.get_account()
    if not account:
        account = self.present_account()
    if lock and account['state'].lower() == 'disabled':
        account = self.enable_account()
    if lock and account['state'].lower() != 'locked' or (not lock and account['state'].lower() != 'disabled'):
        self.result['changed'] = True
        args = {'id': account['id'], 'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'lock': lock}
        if not self.module.check_mode:
            account = self.query_api('disableAccount', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                account = self.poll_job(account, 'account')
    return account