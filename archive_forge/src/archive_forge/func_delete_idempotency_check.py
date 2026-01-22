from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def delete_idempotency_check(module, rest_obj):
    delete_names = module.params['names']
    data = rest_obj.get_all_items_with_pagination(COMPLIANCE_BASELINE)
    available_baseline_map = dict([(item['Id'], item['Name']) for item in data['value']])
    valid_names = set(delete_names) & set(available_baseline_map.values())
    valid_id_list = get_identifiers(available_baseline_map, valid_names)
    if module.check_mode and len(valid_id_list) > 0:
        module.exit_json(msg=CHECK_MODE_CHANGES_MSG, changed=True)
    if len(valid_id_list) == 0:
        module.exit_json(msg=CHECK_MODE_NO_CHANGES_MSG, changed=False)
    return valid_id_list