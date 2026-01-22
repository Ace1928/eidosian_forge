from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def dns_connect(module):
    """ Return an dns connection"""
    dns_params = get_profile(module.params)
    region = module.params.get('alicloud_region')
    if region:
        try:
            dns = connect_to_acs(footmark.dns, region, **dns_params)
        except AnsibleACSError as e:
            module.fail_json(msg=str(e))
    return dns