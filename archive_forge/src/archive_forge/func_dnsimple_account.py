from __future__ import absolute_import, division, print_function
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
def dnsimple_account(self):
    """select a dnsimple account. If a user token is used for authentication,
        this user must only have access to a single account"""
    account = self.client.identity.whoami().data.account
    if not account:
        accounts = Accounts(self.client).list_accounts().data
        if len(accounts) != 1:
            msg = 'The provided dnsimple token is a user token with multiple accounts.Use an account token or a user token with access to a single account.See https://support.dnsimple.com/articles/api-access-token/'
            raise DNSimpleException(msg)
        account = accounts[0]
    self.account = account