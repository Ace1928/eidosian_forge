from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def enhanced_lag_spec():
    return dict(name=dict(type='str', required=True), lacp_mode=dict(type='str', choices=['active', 'passive']), load_balancing_mode=dict(type='str', choices=['dst-ip', 'dst-ip-l4port', 'dst-ip-vlan', 'dst-ip-l4port-vlan', 'dst-mac', 'dst-l4port', 'src-ip', 'src-ip-l4port', 'src-ip-vlan', 'src-ip-l4port-vlan', 'src-mac', 'src-l4port', 'src-dst-ip', 'src-dst-ip-l4port', 'src-dst-ip-vlan', 'src-dst-ip-l4port-vlan', 'src-dst-mac', 'src-dst-l4port', 'src-port-id', 'vlan']), number_uplinks=dict(type='int'))