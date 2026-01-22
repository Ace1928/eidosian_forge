from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def validate_switches_overlap(current_dict, modify_dict, module):
    """
    Validation in case of modify operation when current setting user provided switches details overlaps
    :param current_dict: modify payload created
    :param modify_dict: current payload of specified fabric
    :param module: Ansible module object
    """
    modify_primary_switch = modify_dict.get('PhysicalNode1')
    current_secondary_switch = current_dict.get('PhysicalNode2')
    modify_secondary_switch = modify_dict.get('PhysicalNode2')
    current_primary_switch = current_dict.get('PhysicalNode1')
    if modify_primary_switch and current_primary_switch != modify_primary_switch:
        module.fail_json(msg='The modify operation does not support primary_switch_service_tag update.')
    if modify_secondary_switch and current_secondary_switch != modify_secondary_switch:
        module.fail_json(msg='The modify operation does not support secondary_switch_service_tag update.')
    flag = all([modify_primary_switch, modify_secondary_switch, current_primary_switch, current_secondary_switch]) and (modify_primary_switch == current_secondary_switch and modify_secondary_switch == current_primary_switch)
    if not flag and modify_primary_switch is not None and (current_secondary_switch is not None) and (modify_primary_switch == current_secondary_switch):
        module.fail_json(PRIMARY_SWITCH_OVERLAP_MSG)
    if not flag and modify_secondary_switch is not None and (current_primary_switch is not None) and (modify_secondary_switch == current_primary_switch):
        module.fail_json(SECONDARY_SWITCH_OVERLAP_MSG)