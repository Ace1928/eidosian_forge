from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def check_attr_exists(module, curr_attr, inp_attr):
    invalid_attr = []
    pending_attr = {}
    diff = 0
    for each in inp_attr:
        if each not in curr_attr.keys():
            invalid_attr.append(each)
        elif curr_attr[each] != inp_attr[each]:
            diff = 1
            pending_attr[each] = inp_attr[each]
    if invalid_attr:
        module.exit_json(msg=INVALID_ATTRIBUTES.format(invalid_attr), failed=True)
    if diff and module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    elif not diff:
        module.exit_json(msg=NO_CHANGES_FOUND)
    return pending_attr