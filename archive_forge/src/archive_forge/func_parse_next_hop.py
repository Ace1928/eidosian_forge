from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.static_routes.static_routes import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def parse_next_hop(self, conf):
    nh_list = None
    if conf:
        nh_list = []
        hop_list = findall('^.*next-hop (.+)', conf, M)
        if hop_list:
            for hop in hop_list:
                distance = search('^.*distance (.\\S+)', hop, M)
                interface = search('^.*interface (.\\S+)', hop, M)
                dis = hop.find('disable')
                hop_info = hop.split(' ')
                nh_info = {'forward_router_address': hop_info[0].strip("'")}
                if interface:
                    nh_info['interface'] = interface.group(1).strip("'")
                if distance:
                    value = distance.group(1).strip("'")
                    nh_info['admin_distance'] = int(value)
                elif dis >= 1:
                    nh_info['enabled'] = False
                for element in nh_list:
                    if element['forward_router_address'] == nh_info['forward_router_address']:
                        if 'interface' in nh_info.keys():
                            element['interface'] = nh_info['interface']
                        if 'admin_distance' in nh_info.keys():
                            element['admin_distance'] = nh_info['admin_distance']
                        if 'enabled' in nh_info.keys():
                            element['enabled'] = nh_info['enabled']
                        nh_info = None
                if nh_info is not None:
                    nh_list.append(nh_info)
    return nh_list