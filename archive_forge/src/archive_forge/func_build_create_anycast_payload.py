from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def build_create_anycast_payload(self, commands):
    payload = {}
    if 'anycast_address' in commands and commands['anycast_address']:
        payload = {'sonic-sag:SAG_GLOBAL_LIST': []}
        temp = {}
        if 'ipv4' in commands['anycast_address'] and commands['anycast_address']['ipv4']:
            temp.update({'IPv4': 'enable'})
        if 'ipv4' in commands['anycast_address'] and (not commands['anycast_address']['ipv4']):
            temp.update({'IPv4': 'disable'})
        if 'ipv6' in commands['anycast_address'] and commands['anycast_address']['ipv6']:
            temp.update({'IPv6': 'enable'})
        if 'ipv6' in commands['anycast_address'] and (not commands['anycast_address']['ipv6']):
            temp.update({'IPv6': 'disable'})
        if 'mac_address' in commands['anycast_address'] and commands['anycast_address']['mac_address']:
            temp.update({'gwmac': commands['anycast_address']['mac_address']})
        if temp:
            temp.update({'table_distinguisher': 'IP'})
            payload['sonic-sag:SAG_GLOBAL_LIST'].append(temp)
    return payload