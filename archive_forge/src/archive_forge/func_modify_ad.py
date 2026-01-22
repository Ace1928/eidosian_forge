from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def modify_ad(module, rest_obj, ad):
    prm = module.params
    modify_payload = make_payload(prm)
    ad = rest_obj.strip_substr_dict(ad)
    if ad.get('ServerName'):
        ad.get('ServerName').sort()
    if modify_payload.get('ServerName'):
        modify_payload.get('ServerName').sort()
    diff = recursive_diff(modify_payload, ad)
    is_change = False
    if diff:
        if diff[0]:
            is_change = True
            ad.update(modify_payload)
    msg = validate_n_testconnection(module, rest_obj, ad)
    if not is_change and (not ad.get('CertificateValidation')):
        module.exit_json(msg='{0}{1}'.format(msg, NO_CHANGES_MSG), active_directory=ad)
    if module.check_mode:
        module.exit_json(msg='{0}{1}'.format(msg, CHANGES_FOUND), changed=True)
    resp = rest_obj.invoke_request('PUT', '{0}({1})'.format(AD_URI, ad['Id']), data=ad)
    ad = resp.json_data
    ad.pop('CertificateFile', '')
    module.exit_json(msg='{0}{1}'.format(msg, MODIFY_SUCCESS), active_directory=ad, changed=True)