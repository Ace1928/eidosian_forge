from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def process_check_mode(module, diff):
    if not diff:
        module.exit_json(msg=NO_CHANGES_MSG)
    elif diff and module.check_mode:
        module.exit_json(msg=CHANGES_MSG, changed=True)