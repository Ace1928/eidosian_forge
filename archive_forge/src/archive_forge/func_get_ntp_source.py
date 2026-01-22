from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_ntp_source(module):
    source_type = None
    source = None
    command = 'show run | inc ntp.source'
    output = execute_show_command(command, module, command_type='cli_show_ascii')
    if output:
        try:
            if 'interface' in output[0]:
                source_type = 'source-interface'
            else:
                source_type = 'source'
            source = output[0].split()[2].lower()
        except (AttributeError, IndexError):
            source_type = None
            source = None
    return (source_type, source)