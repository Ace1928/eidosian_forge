from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def communities_tmplt(config_data):
    name = config_data.get('name', '')
    command = 'snmp-server community {name}'.format(name=name)
    if config_data.get('rw'):
        command += ' RW'
    elif config_data.get('ro'):
        command += ' RO'
    if config_data.get('sdrowner'):
        command += ' SDROwner'
    elif config_data.get('systemowner'):
        command += ' SystemOwner'
    if config_data.get('acl_v4'):
        command += ' IPv4 {IPv4}'.format(IPv4=config_data['acl_v4'])
    if config_data.get('acl_v6'):
        command += ' IPv6 {IPv6}'.format(IPv6=config_data['acl_v6'])
    if config_data.get('v4_acl'):
        command += ' {v4_acl}'.format(v4_acl=config_data['v4_acl'])
    return command