from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import run_commands, get_config, load_config, get_diff
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def get_startup_config(module, flags=None):
    reply = run_commands(module, {'command': 'show switch', 'output': 'text'})
    match = re.search('Config Selected: +(\\S+)\\.cfg', to_text(reply, errors='surrogate_or_strict').strip(), re.MULTILINE)
    if match:
        cfgname = match.group(1).strip()
        command = ' '.join(['debug cfgmgr show configuration file', cfgname])
        if flags:
            command += ' '.join(to_list(flags)).strip()
        reply = run_commands(module, {'command': command, 'output': 'text'})
        data = reply[0]
    else:
        data = ''
    return data