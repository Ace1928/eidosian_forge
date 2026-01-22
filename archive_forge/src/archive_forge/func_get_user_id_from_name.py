from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_user_id_from_name(rest_obj, name):
    """Get the account id using account name"""
    user_id = None
    if name is not None:
        resp = rest_obj.invoke_request('GET', 'AccountService/Accounts')
        if resp.success:
            for user in resp.json_data.get('value'):
                if 'UserName' in user and user['UserName'] == name:
                    return user['Id']
    return user_id