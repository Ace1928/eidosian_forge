from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def idempotency_check_for_state_present(fabric_id, current_payload, expected_payload, module):
    """
    idempotency check in case of state present
    :param fabric_id: fabric id
    :param current_payload: payload created
    :param expected_payload: already existing payload for specified fabric
    :param module: ansible module object
    :return: None
    """
    if fabric_id:
        exp_dict = expected_payload.copy()
        cur_dict = current_payload.copy()
        for d in (exp_dict, cur_dict):
            fab_dz_lst = d.pop('FabricDesignMapping', [])
            for fab in fab_dz_lst:
                d[fab.get('DesignNode')] = fab.get('PhysicalNode')
        payload_diff = compare_payloads(exp_dict, cur_dict)
        if module.check_mode:
            if payload_diff:
                module.exit_json(msg=CHECK_MODE_CHANGE_FOUND_MSG, changed=True)
            else:
                module.exit_json(msg=CHECK_MODE_CHANGE_NOT_FOUND_MSG, changed=False)
        elif not payload_diff:
            module.exit_json(msg=IDEMPOTENCY_MSG, changed=False)
    elif module.check_mode:
        module.exit_json(msg=CHECK_MODE_CHANGE_FOUND_MSG, changed=True)