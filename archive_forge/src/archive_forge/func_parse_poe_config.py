from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def parse_poe_config(poe, power):
    if poe.get('by_class') is not None:
        power += 'power-by-class %s' % poe.get('by_class')
    elif poe.get('limit') is not None:
        power += 'power-limit %s' % poe.get('limit')
    elif poe.get('priority') is not None:
        power += 'priority %s' % poe.get('priority')
    elif poe.get('enabled'):
        power = 'inline power'
    elif poe.get('enabled') is False:
        power = 'no inline power'
    return power