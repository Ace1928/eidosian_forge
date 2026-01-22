from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import check_sdk
def get_required_parameter(param, env_var, required=False):
    var = params.get(param) or os.environ.get(env_var)
    if not var and required and (state == 'present'):
        module.fail_json(msg="'%s' is a required parameter." % param)
    return var