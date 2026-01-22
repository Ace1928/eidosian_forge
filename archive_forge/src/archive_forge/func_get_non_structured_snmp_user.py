from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_non_structured_snmp_user(body_text):
    resource = {}
    output = body_text.rsplit('__________')[-1]
    pat = re.compile('^(?P<user>\\S+)\\s+(?P<auth>\\S+)\\s+(?P<priv>[\\w\\d-]+)(?P<enforce>\\([\\w\\d-]+\\))*\\s+(?P<group>\\S+)', re.M)
    m = re.search(pat, output)
    if not m:
        return resource
    resource['user'] = m.group('user')
    resource['auth'] = m.group('auth')
    resource['encrypt'] = 'aes-128' if 'aes' in str(m.group('priv')) else 'none'
    resource['group'] = [m.group('group')]
    more_groups = re.findall('^\\s+([\\w\\d-]+)\\s*$', output, re.M)
    if more_groups:
        resource['group'] += more_groups
    return resource