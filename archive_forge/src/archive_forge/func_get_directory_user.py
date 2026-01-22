from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_directory_user(module, rest_obj):
    user_group_name, user = (module.params.get('group_name'), None)
    state = module.params['state']
    if user_group_name is None:
        module.fail_json(msg='missing required arguments: group_name')
    user_resp = rest_obj.invoke_request('GET', ACCOUNT_URI)
    for usr in user_resp.json_data['value']:
        if usr['UserName'].lower() == user_group_name.lower() and usr['UserTypeId'] == 2:
            user = usr
            if module.check_mode and state == 'absent':
                user = rest_obj.strip_substr_dict(usr)
                module.exit_json(msg=CHANGES_FOUND, changed=True, domain_user_status=user)
            break
    else:
        if state == 'absent':
            module.exit_json(msg=NO_CHANGES_MSG)
    return user