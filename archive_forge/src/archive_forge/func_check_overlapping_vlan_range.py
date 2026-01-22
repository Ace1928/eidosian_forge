from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def check_overlapping_vlan_range(payload, vlans):
    current_vlan = None
    for xtype in vlans:
        overlap = list(range(max(xtype.get('VlanMinimum', 0), payload['VlanMinimum']), min(xtype.get('VlanMaximum', 0), payload['VlanMaximum']) + 1))
        if overlap:
            current_vlan = xtype
            break
    return current_vlan