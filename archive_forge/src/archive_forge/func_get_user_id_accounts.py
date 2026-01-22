from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_user_id_accounts(idrac, module, accounts_uri, user_id):
    acc_dets_json_data = {}
    try:
        acc_uri = accounts_uri + '/{0}'.format(user_id)
        acc_dets = idrac.invoke_request(acc_uri, 'GET')
        acc_dets_json_data = strip_substr_dict(acc_dets.json_data)
        if acc_dets_json_data.get('Oem') is not None:
            acc_dets_json_data['Oem']['Dell'] = strip_substr_dict(acc_dets_json_data['Oem']['Dell'])
        acc_dets_json_data.pop('Links', None)
    except HTTPError:
        module.exit_json(msg=INVALID_USERID, failed=True)
    return acc_dets_json_data