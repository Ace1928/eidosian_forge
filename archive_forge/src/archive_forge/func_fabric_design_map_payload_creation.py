from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def fabric_design_map_payload_creation(design_map_modify_payload, design_map_current_payload, module):
    """
    process FabricDesignMapping contents
    :param design_map_modify_payload: modify payload created
    :param design_map_current_payload: current payload of specified fabric
    :param module: Ansible module object
    :return: list
    """
    modify_dict = design_node_dict_update(design_map_modify_payload)
    current_dict = design_node_dict_update(design_map_current_payload)
    validate_switches_overlap(current_dict, modify_dict, module)
    current_dict.update(modify_dict)
    design_list = []
    for key, val in current_dict.items():
        if key == 'PhysicalNode1':
            design_list.append({'DesignNode': 'Switch-A', 'PhysicalNode': val})
        else:
            design_list.append({'DesignNode': 'Switch-B', 'PhysicalNode': val})
    return design_list