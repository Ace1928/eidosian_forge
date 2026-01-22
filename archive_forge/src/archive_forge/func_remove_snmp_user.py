from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def remove_snmp_user(user, group=None):
    if group:
        return ['no snmp-server user {0} {1}'.format(user, group)]
    else:
        return ['no snmp-server user {0}'.format(user)]