from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_access_group_vrfs(config_data):
    commands = []
    vrf_name = config_data.get('name')
    base_command = 'ntp access-group vrf {name}'.format(name=vrf_name)
    for ip in ['ipv4', 'ipv6']:
        if config_data.get(ip, {}).get('serve'):
            commands.append('{base_command} {ip} serve {serve}'.format(base_command=base_command, serve=config_data.get(ip, {}).get('serve'), ip=ip))
        if config_data.get(ip, {}).get('serve_only'):
            commands.append('{base_command} {ip} serve-only {serve_only}'.format(base_command=base_command, serve_only=config_data.get(ip, {}).get('serve_only'), ip=ip))
        if config_data.get(ip, {}).get('query_only'):
            commands.append('{base_command} {ip} query-only {query_only}'.format(base_command=base_command, query_only=config_data.get(ip, {}).get('query_only'), ip=ip))
        if config_data.get(ip, {}).get('peer'):
            commands.append('{base_command} {ip} peer {peer}'.format(base_command=base_command, peer=config_data.get(ip, {}).get('peer'), ip=ip))
    return commands