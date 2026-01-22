from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, ConfigLine
def get_sublevel_config(running_config, module):
    contents = list()
    current_config_contents = list()
    running_config = NetworkConfig(contents=running_config, indent=1)
    obj = running_config.get_object(module.params['parents'])
    if obj:
        contents = obj.children
    parents = module.params['parents']
    if parents[2:]:
        temp = 1
        for count, item in enumerate(parents[2:], start=2):
            item = ' ' * temp + item
            temp = temp + 1
            parents[count] = item
    contents[:0] = parents
    indent = 0
    for c in contents:
        if isinstance(c, str):
            if c in parents:
                current_config_contents.append(c.rjust(len(c) + indent, ' '))
            if c not in parents:
                c = ' ' * (len(parents) - 1) + c
                current_config_contents.append(c.rjust(len(c) + indent, ' '))
        if isinstance(c, ConfigLine):
            current_config_contents.append(c.raw)
        indent = 1
    sublevel_config = '\n'.join(current_config_contents)
    return sublevel_config