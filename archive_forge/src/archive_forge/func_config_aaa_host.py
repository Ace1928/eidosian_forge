from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_aaa_host(server_type, address, params, existing):
    cmds = []
    cmd_str = '{0}-server host {1}'.format(server_type, address)
    cmd_no_str = 'no ' + cmd_str
    key = params.get('key')
    enc_type = params.get('encrypt_type', '')
    defval = False
    nondef = False
    if key:
        if key != 'default':
            cmds.append(cmd_str + ' key {0} {1}'.format(enc_type, key))
        else:
            cmds.append(cmd_no_str + ' key 7 {0}'.format(existing.get('key')))
    locdict = {'auth_port': 'auth-port', 'acct_port': 'acct-port', 'tacacs_port': 'port', 'host_timeout': 'timeout'}
    for key in ['auth_port', 'acct_port', 'tacacs_port', 'host_timeout']:
        item = params.get(key)
        if item:
            if item != 'default':
                cmd_str += ' {0} {1}'.format(locdict.get(key), item)
                nondef = True
            else:
                cmd_no_str += ' {0} 1'.format(locdict.get(key))
                defval = True
    if defval:
        cmds.append(cmd_no_str)
    if nondef or not existing:
        cmds.append(cmd_str)
    return cmds