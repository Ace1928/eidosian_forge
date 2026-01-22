from __future__ import absolute_import, division, print_function
import re
import time
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_import(configobj, name):
    cfg = configobj['vrf definition %s' % name]
    cfg = '\n'.join(cfg.children)
    matches = re.findall('route-target\\s+import\\s+(.+)', cfg, re.M)
    return matches