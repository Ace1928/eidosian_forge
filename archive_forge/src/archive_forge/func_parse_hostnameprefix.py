from __future__ import absolute_import, division, print_function
import collections
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def parse_hostnameprefix(self, line):
    prefix = None
    match = re.search('logging hostnameprefix (\\S+)', line, re.M)
    if match:
        prefix = match.group(1)
    return prefix