from __future__ import absolute_import, division, print_function
import re
from time import sleep
import itertools
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def spanning_tree(module, stp):
    stp_cmd = list()
    if stp.get('enabled') is False:
        if stp.get('type') == '802-1w':
            stp_cmd.append('no spanning-tree' + ' ' + stp.get('type'))
        stp_cmd.append('no spanning-tree')
    elif stp.get('type'):
        stp_cmd.append('spanning-tree' + ' ' + stp.get('type'))
        if stp.get('priority') and stp.get('type') == 'rstp':
            module.fail_json(msg='spanning-tree 802-1w only can have priority')
        elif stp.get('priority'):
            stp_cmd.append('spanning-tree' + ' ' + stp.get('type') + ' ' + 'priority' + ' ' + stp.get('priority'))
    return stp_cmd