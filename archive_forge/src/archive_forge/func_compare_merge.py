from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def compare_merge(module, attr_dict):
    val_map = {'ip_range': 'LoginSecurity.1#IPRangeAddr', 'enable_ip_range': 'LoginSecurity.1#IPRangeEnable', 'by_ip_address': 'LoginSecurity.1#LockoutByIPEnable', 'by_user_name': 'LoginSecurity.1#LockoutByUsernameEnable', 'lockout_fail_count': 'LoginSecurity.1#LockoutFailCount', 'lockout_fail_window': 'LoginSecurity.1#LockoutFailCountTime', 'lockout_penalty_time': 'LoginSecurity.1#LockoutPenaltyTime'}
    diff = 0
    inp_dicts = ['restrict_allowed_ip_range', 'login_lockout_policy']
    for d in inp_dicts:
        inp_dict = module.params.get(d, {})
        if inp_dict:
            for k, v in inp_dict.items():
                if v is not None:
                    if attr_dict[val_map[k]] != v:
                        attr_dict[val_map[k]] = v
                        diff = diff + 1
    if attr_dict.get('LoginSecurity.1#IPRangeEnable') is False:
        if attr_dict.get('LoginSecurity.1#IPRangeAddr') is not None:
            attr_dict['LoginSecurity.1#IPRangeAddr'] = None
            diff = diff - 1
    if not diff:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    return attr_dict