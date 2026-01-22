from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_config, load_config
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList, validate_ip_v6_address
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
def parse_aaa_servers(config):
    configlines = config.split('\n')
    obj = []
    for line in configlines:
        auth_key_type = []
        if 'radius-server' in line or 'tacacs-server' in line:
            aaa_type = 'radius' if 'radius-server' in line else 'tacacs'
            match = re.search('(host ipv6 (\\S+))|(host (\\S+))', line)
            if match:
                hostname = match.group(2) if match.group(2) is not None else match.group(4)
            match = re.search('auth-port ([0-9]+)', line)
            if match:
                auth_port_num = match.group(1)
            else:
                auth_port_num = None
            match = re.search('acct-port ([0-9]+)', line)
            if match:
                acct_port_num = match.group(1)
            else:
                acct_port_num = None
            match = re.search('acct-port [0-9]+ (\\S+)', line)
            if match:
                acct_type = match.group(1)
            else:
                acct_type = None
            if aaa_type == 'tacacs':
                match = re.search('auth-port [0-9]+ (\\S+)', line)
                if match:
                    acct_type = match.group(1)
                else:
                    acct_type = None
            match = re.search('(dot1x)', line)
            if match:
                auth_key_type.append('dot1x')
            match = re.search('(mac-auth)', line)
            if match:
                auth_key_type.append('mac-auth')
            match = re.search('(web-auth)', line)
            if match:
                auth_key_type.append('web-auth')
            obj.append({'type': aaa_type, 'hostname': hostname, 'auth_port_type': 'auth-port', 'auth_port_num': auth_port_num, 'acct_port_num': acct_port_num, 'acct_type': acct_type, 'auth_key': None, 'auth_key_type': set(auth_key_type) if len(auth_key_type) > 0 else None})
    return obj