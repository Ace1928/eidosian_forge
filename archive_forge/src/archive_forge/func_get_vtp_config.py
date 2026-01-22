from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_vtp_config(module):
    command = 'show vtp status'
    body = execute_show_command(command, module, 'text')[0]
    vtp_parsed = {}
    if body:
        version_regex = '.*VTP version running\\s+:\\s+(?P<version>\\d).*'
        domain_regex = '.*VTP Domain Name\\s+:\\s+(?P<domain>\\S+).*'
        try:
            match_version = re.match(version_regex, body, re.DOTALL)
            version = match_version.groupdict()['version']
        except AttributeError:
            version = ''
        try:
            match_domain = re.match(domain_regex, body, re.DOTALL)
            domain = match_domain.groupdict()['domain']
        except AttributeError:
            domain = ''
        if domain and version:
            vtp_parsed['domain'] = domain
            vtp_parsed['version'] = version
            vtp_parsed['vtp_password'] = get_vtp_password(module)
    return vtp_parsed