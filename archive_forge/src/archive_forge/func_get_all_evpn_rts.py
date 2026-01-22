from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_all_evpn_rts(self, evpn_rts):
    """Get all EVPN RTS"""
    rts = evpn_rts.findall('evpnRT')
    if not rts:
        return
    for ele in rts:
        vrf_rttype = ele.find('vrfRTType')
        vrf_rtvalue = ele.find('vrfRTValue')
        if vrf_rttype.text == 'export_extcommunity':
            self.evpn_info['vpn_target_export'].append(vrf_rtvalue.text)
        elif vrf_rttype.text == 'import_extcommunity':
            self.evpn_info['vpn_target_import'].append(vrf_rtvalue.text)