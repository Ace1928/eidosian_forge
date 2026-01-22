from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_payload_from_parameters(params):
    payload = {}
    for parameter in params:
        parameter_value = params[parameter]
        if parameter_value is not None and is_checkpoint_param(parameter):
            if isinstance(parameter_value, dict):
                payload[parameter.replace('_', '-')] = get_payload_from_parameters(parameter_value)
            elif isinstance(parameter_value, list) and len(parameter_value) != 0 and isinstance(parameter_value[0], dict):
                payload_list = []
                for element_dict in parameter_value:
                    payload_list.append(get_payload_from_parameters(element_dict))
                payload[parameter.replace('_', '-')] = payload_list
            else:
                if parameter == 'gateway_version' or parameter == 'cluster_version' or parameter == 'server_version' or (parameter == 'check_point_host_version') or (parameter == 'target_version') or (parameter == 'vsx_version'):
                    parameter = 'version'
                elif parameter == 'login_message':
                    parameter = 'message'
                payload[parameter.replace('_', '-')] = parameter_value
    return payload