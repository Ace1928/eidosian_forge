from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_breakout_payload(device_id, breakout_type, interface_id):
    """
    Payload for breakout configuration.
    :param device_id: device id
    :param breakout_type: requested breakout type
    :param interface_id: port number with service tag
    :return: json
    """
    payload = {'Id': 0, 'JobName': 'Breakout Port', 'JobDescription': '', 'Schedule': 'startnow', 'State': 'Enabled', 'JobType': {'Id': 3, 'Name': 'DeviceAction_Task'}, 'Params': [{'Key': 'breakoutType', 'Value': breakout_type}, {'Key': 'interfaceId', 'Value': interface_id}, {'Key': 'operationName', 'Value': 'CONFIGURE_PORT_BREAK_OUT'}], 'Targets': [{'JobId': 0, 'Id': device_id, 'Data': '', 'TargetType': {'Id': 4000, 'Name': 'DEVICE'}}]}
    return payload