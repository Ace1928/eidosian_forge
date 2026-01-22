from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.vlans.vlans import (
def parse_vlan_config(self, vlan_conf):
    vlan_list = list()
    re1 = re.compile('^vlan configuration +(?P<vlan>\\d+)$')
    re2 = re.compile('^member +(evpn\\-instance +(?P<evi>\\d+) )?vni (?P<vni>[\\d\\-]+)$')
    for line in vlan_conf.splitlines():
        line = line.strip()
        m = re1.match(line)
        if m:
            vlan = m.groupdict()['vlan']
            vlan_dict = {'vlan_id': vlan}
            continue
        m = re2.match(line)
        if m:
            group = m.groupdict()
            vlan_dict.update({'member': {}})
            vlan_dict['member'].update({'vni': group['vni']})
            if group['evi']:
                vlan_dict['member'].update({'evi': group['evi']})
            vlan_list.append(vlan_dict)
    return vlan_list