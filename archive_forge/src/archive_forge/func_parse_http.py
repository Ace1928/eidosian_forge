from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import Version
def parse_http(data):
    http_res = ['nxapi http port (\\d+)']
    http_port = None
    for regex in http_res:
        match = re.search(regex, data, re.M)
        if match:
            http_port = int(match.group(1))
            break
    return {'http': http_port is not None, 'http_port': http_port}