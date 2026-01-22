from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_tls_servers(config_data):
    commands = []
    name = config_data.get('name')
    base_command = 'logging tls-server {name}'.format(name=name)
    if config_data.get('tls_hostname'):
        commands.append('{base_command} tls-hostname {tls}'.format(tls=config_data['tls_hostname'], base_command=base_command))
    if config_data.get('trustpoint'):
        commands.append('{base_command} trustpoint {trustpoint}'.format(trustpoint=config_data['trustpoint'], base_command=base_command))
    if config_data.get('vrf'):
        commands.append('{base_command} vrf {vrf}'.format(vrf=config_data['vrf'], base_command=base_command))
    if config_data.get('severity'):
        commands.append('{base_command} severity {severity}'.format(severity=config_data['severity'], base_command=base_command))
    if len(commands) == 0:
        commands.append(base_command)
    return commands