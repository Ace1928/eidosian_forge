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
def parse_ports(interfaces, ports, lags):
    for interface in interfaces:
        low, high = extract_list_from_interface(interface)
        while high >= low:
            if 'ethernet' in interface:
                if not low in ports:
                    module.fail_json(msg='One or more conditional statements have not been satisfied ' + interface)
            if 'lag' in interface:
                if not low in lags:
                    module.fail_json(msg='One or more conditional statements have not been satisfied ' + interface)
            low = low + 1