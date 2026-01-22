from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def handle_handler_api_and_type(payload_handler):
    if payload_handler['type'] in ['tcp_stream_handler', 'sumo_logic_metrics_handler']:
        payload_handler['api_version'] = API_VERSION['v1']
    else:
        payload_handler['api_version'] = API_VERSION['v2']
    payload_handler['type'] = HANDLER_TYPE[payload_handler['type']]