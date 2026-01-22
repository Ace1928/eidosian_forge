from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_snmp_host(delta, udp, proposed, existing, module):
    commands = []
    command_builder = []
    host = proposed['snmp_host']
    cmd = 'snmp-server host {0}'.format(proposed['snmp_host'])
    snmp_type = delta.get('snmp_type')
    version = delta.get('version')
    ver = delta.get('v3')
    community = delta.get('community')
    command_builder.append(cmd)
    if any([snmp_type, version, ver, community]):
        type_string = snmp_type or existing.get('type')
        if type_string:
            command_builder.append(type_string)
        version = version or existing.get('version')
        if version:
            if version == 'v1':
                vn = '1'
            elif version == 'v2c':
                vn = '2c'
            elif version == 'v3':
                vn = '3'
            version_string = 'version {0}'.format(vn)
            command_builder.append(version_string)
        if ver:
            ver_string = ver or existing.get('v3')
            command_builder.append(ver_string)
        if community:
            community_string = community or existing.get('community')
            command_builder.append(community_string)
        udp_string = ' udp-port {0}'.format(udp)
        command_builder.append(udp_string)
        cmd = ' '.join(command_builder)
        commands.append(cmd)
    CMDS = {'vrf_filter': 'snmp-server host {0} filter-vrf {vrf_filter} udp-port {1}', 'vrf': 'snmp-server host {0} use-vrf {vrf} udp-port {1}', 'src_intf': 'snmp-server host {0} source-interface {src_intf} udp-port {1}'}
    for key in delta:
        command = CMDS.get(key)
        if command:
            cmd = command.format(host, udp, **delta)
            commands.append(cmd)
    return commands