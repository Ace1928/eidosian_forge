from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_route_target_value(arg, config, module):
    splitted_config = config.splitlines()
    value_list = []
    command = PARAM_TO_COMMAND_KEYMAP.get(arg)
    command_re = re.compile('(?:{0}\\s)(?P<value>.*)$'.format(command), re.M)
    for line in splitted_config:
        value = ''
        if command in line.strip():
            value = command_re.search(line).group('value')
            value_list.append(value)
    return value_list