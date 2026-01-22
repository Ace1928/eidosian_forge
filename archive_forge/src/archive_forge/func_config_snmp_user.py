from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_snmp_user(proposed, user, reset):
    if reset:
        commands = remove_snmp_user(user)
    else:
        commands = []
    if proposed.get('group'):
        cmd = 'snmp-server user {0} {group}'.format(user, **proposed)
    else:
        cmd = 'snmp-server user {0}'.format(user)
    auth = proposed.get('authentication', None)
    pwd = proposed.get('pwd', None)
    if auth and pwd:
        cmd += ' auth {authentication} {pwd}'.format(**proposed)
    encrypt = proposed.get('encrypt', None)
    privacy = proposed.get('privacy', None)
    if encrypt and privacy:
        cmd += ' priv {encrypt} {privacy}'.format(**proposed)
    elif privacy:
        cmd += ' priv {privacy}'.format(**proposed)
    if cmd:
        commands.append(cmd)
    return commands