from __future__ import absolute_import, division, print_function
import traceback
import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def build_interfaces_create_request(interface_name):
    url = 'data/openconfig-interfaces:interfaces'
    method = 'PATCH'
    payload_template = '{"openconfig-interfaces:interfaces": {"interface": [{"name": "{{interface_name}}", "config": {"name": "{{interface_name}}"}}]}}'
    input_data = {'interface_name': interface_name}
    env = jinja2.Environment(autoescape=False)
    t = env.from_string(payload_template)
    intended_payload = t.render(input_data)
    ret_payload = json.loads(intended_payload)
    request = {'path': url, 'method': method, 'data': ret_payload}
    return request