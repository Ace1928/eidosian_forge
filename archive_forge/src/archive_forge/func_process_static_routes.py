from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.static_routes.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def process_static_routes(self, objs):

    def update_netmask_to_cidr(address, netmask):
        dest = address + '/' + netmask_to_cidr(netmask)
        return dest
    strout = {}
    for k, obj in objs.items():
        _routes = {'next_hops': []}
        _nx_hop = []
        is_vrf = False
        for routes in obj:
            _vrf = routes.pop('_vrf', None)
            if _vrf:
                is_vrf = True
            _afi = routes.pop('_afi')
            _dest = routes.pop('_dest')
            _topology = routes.pop('_topology', None)
            _netmask = routes.pop('_netmask', None)
            _routes['dest'] = update_netmask_to_cidr(_dest, _netmask) if _afi == 'ipv4' else _dest
            if _topology:
                _routes['topology'] = _topology
            _nx_hop.append(routes)
        _routes['next_hops'].extend(_nx_hop)
        if is_vrf:
            if strout.get(_vrf) and strout[_vrf].get(_afi):
                strout[_vrf][_afi].append(_routes)
            elif strout.get(_vrf):
                _tma = {_afi: [_routes]}
                strout[_vrf].update(_tma)
            else:
                _tm = {_vrf: {_afi: [_routes]}}
                strout.update(_tm)
        elif strout.get(_afi):
            strout[_afi].append(_routes)
        else:
            _tma = {_afi: [_routes]}
            strout.update(_tma)
    return strout