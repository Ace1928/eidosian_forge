from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.arista.eos.plugins.module_utils.network.eos.argspec.vlans.vlans import (
def vlan_to_list(vlan_str):
    vlans = []
    for vlan in vlan_str.split(','):
        if '-' in vlan:
            start, stop = vlan.split('-')
            vlans.extend(range(int(start), int(stop) + 1))
        else:
            vlans.append(int(vlan))
    return vlans