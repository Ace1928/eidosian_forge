from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_account_info(self, rest_api, headers=None):
    """
        Get Account
        :return: Account ID
        """
    headers = {'X-User-Token': rest_api.token_type + ' ' + rest_api.token}
    api = '/tenancy/account'
    account_res, error, dummy = rest_api.get(api, header=headers)
    if error is not None:
        return (None, error)
    return (account_res, None)