from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.l2_interfaces.l2_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.l2_interfaces import (
def process_vlans(obj, vlan_type):
    vlans = []
    _vlans = obj.get('trunk')
    if _vlans.get(vlan_type):
        vlans.extend(_vlans.get(vlan_type))
    if _vlans.get(vlan_type + '_add'):
        for vlan_grp in _vlans.get(vlan_type + '_add'):
            vlans.extend(vlan_grp)
        del _vlans[vlan_type + '_add']
    return vlans