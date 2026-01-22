from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def match_current_rt(rt, direction, current, rt_commands):
    command = 'route-target %s %s' % (direction, rt.get('rt'))
    match = re.findall(command, current, re.M)
    want = bool(rt.get('state') != 'absent')
    if not match and want:
        rt_commands.append(command)
    elif match and (not want):
        rt_commands.append('no %s' % command)
    return rt_commands