from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def create_modify_fabric(name, all_fabric, rest_obj, module):
    """
    fabric management actions creation/update of smart fabric
    :param all_fabric: all available fabrics in system
    :param rest_obj: current session object
    :param module: ansible module object
    :param name: fabric name specified
    :return: None
    """
    fabric_id, current_fabric_details = get_fabric_id_details(name, all_fabric)
    required_field_check_for_create(fabric_id, module)
    host_service_tag, msm_version = get_msm_device_details(rest_obj, module)
    validate_devices(host_service_tag, rest_obj, module)
    uri = FABRIC_URI
    expected_payload = create_modify_payload(module.params, fabric_id, msm_version)
    payload = dict(expected_payload)
    method = 'POST'
    msg = 'Fabric creation operation is initiated.'
    current_payload = {}
    if fabric_id:
        current_payload = get_current_payload(current_fabric_details, rest_obj)
        validate_modify(module, current_payload)
        method = 'PUT'
        msg = 'Fabric modification operation is initiated.'
        uri = FABRIC_ID_URI.format(fabric_id=fabric_id)
        payload = merge_payload(expected_payload, current_payload, module)
    idempotency_check_for_state_present(fabric_id, current_payload, expected_payload, module)
    resp = rest_obj.invoke_request(method, uri, data=payload)
    fabric_resp = resp.json_data
    process_output(name, fabric_resp, msg, fabric_id, rest_obj, module)