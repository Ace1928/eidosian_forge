from __future__ import absolute_import, division, print_function
import re
import socket
import sys
import traceback
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import exec_command, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import NetconfConnection
def rm_config_prefix(cfg):
    if not cfg:
        return cfg
    cmds = cfg.split('\n')
    for i in range(len(cmds)):
        if not cmds[i]:
            continue
        if '~' in cmds[i]:
            index = cmds[i].index('~')
            if cmds[i][:index] == ' ' * index:
                cmds[i] = cmds[i].replace('~', '', 1)
    return '\n'.join(cmds)