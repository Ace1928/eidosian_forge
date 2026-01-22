from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def build_install_cmd_set(issu, image, kick, type, force=True):
    commands = ['terminal dont-ask']
    if re.search('required|desired|yes', issu):
        if kick is None:
            issu_cmd = 'non-disruptive'
        else:
            issu_cmd = ''
    elif kick is None:
        issu_cmd = ''
    else:
        issu_cmd = 'force' if force else ''
    if type == 'impact':
        rootcmd = 'show install all impact'
        if kick:
            issu_cmd = ''
    else:
        rootcmd = 'install all'
    if kick is None:
        commands.append('%s nxos %s %s' % (rootcmd, image, issu_cmd))
    else:
        commands.append('%s %s system %s kickstart %s' % (rootcmd, issu_cmd, image, kick))
    return commands