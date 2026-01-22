from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def parse_shutdown(configobj, name):
    cfg = configobj['interface %s' % name]
    cfg = '\n'.join(cfg.children)
    match = re.search('^shutdown', cfg, re.M)
    if match:
        return True
    else:
        return False