from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.system.system import SystemArgs
def parse_sonic_system(self, spec, conf):
    config = deepcopy(spec)
    if conf:
        if 'hostname' in conf and conf['hostname']:
            config['hostname'] = conf['hostname']
        if 'intf_naming_mode' in conf and conf['intf_naming_mode']:
            config['interface_naming'] = conf['intf_naming_mode']
        if 'IPv4' in conf and conf['IPv4'] == 'enable':
            config['anycast_address']['ipv4'] = True
        if 'IPv4' in conf and conf['IPv4'] == 'disable':
            config['anycast_address']['ipv4'] = False
        if 'IPv6' in conf and conf['IPv6'] == 'enable':
            config['anycast_address']['ipv6'] = True
        if 'IPv6' in conf and conf['IPv6'] == 'disable':
            config['anycast_address']['ipv6'] = False
        if 'gwmac' in conf and conf['gwmac']:
            config['anycast_address']['mac_address'] = conf['gwmac']
    return utils.remove_empties(config)