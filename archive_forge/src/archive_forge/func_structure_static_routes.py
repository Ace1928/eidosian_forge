from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.static_routes.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def structure_static_routes(self, strout):
    _static_route_facts = []
    afi_v4 = strout.pop('ipv4', None)
    afi_v6 = strout.pop('ipv6', None)
    if afi_v4 or afi_v6:
        _triv_static_route = {'address_families': []}
        if afi_v4:
            _triv_static_route['address_families'].append({'afi': 'ipv4', 'routes': afi_v4})
        if afi_v6:
            _triv_static_route['address_families'].append({'afi': 'ipv6', 'routes': afi_v6})
        _static_route_facts.append(_triv_static_route)
    for k, v in strout.items():
        afi_v4 = v.pop('ipv4', None)
        afi_v6 = v.pop('ipv6', None)
        _vrf_static_route = {'vrf': k, 'address_families': []}
        if afi_v4:
            _vrf_static_route['address_families'].append({'afi': 'ipv4', 'routes': afi_v4})
        if afi_v6:
            _vrf_static_route['address_families'].append({'afi': 'ipv6', 'routes': afi_v6})
        _static_route_facts.append(_vrf_static_route)
    return _static_route_facts