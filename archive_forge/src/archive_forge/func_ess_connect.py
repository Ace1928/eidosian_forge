from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def ess_connect(module):
    """ Return an ess connection"""
    ess_params = get_profile(module.params)
    region = module.params.get('alicloud_region')
    if region:
        try:
            ess = connect_to_acs(footmark.ess, region, **ess_params)
        except AnsibleACSError as e:
            module.fail_json(msg=str(e))
    return ess