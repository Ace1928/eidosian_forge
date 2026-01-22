from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def fips_mode_enable(module, rest_obj):
    resp = rest_obj.invoke_request('GET', FIPS_MODE)
    fips_payload = resp.json_data
    curr_fips_mode = fips_payload.get('FipsMode')
    if module.params.get('fips_mode_enable') is True:
        fips_mode = 'ON'
    else:
        fips_mode = 'OFF'
    if curr_fips_mode.lower() == fips_mode.lower():
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    payload = rest_obj.strip_substr_dict(fips_payload)
    payload['FipsMode'] = fips_mode
    rest_obj.invoke_request('PUT', FIPS_MODE, data=payload)
    module.exit_json(msg=FIPS_TOGGLED.format('disabled' if fips_mode == 'OFF' else 'enabled'), changed=True)