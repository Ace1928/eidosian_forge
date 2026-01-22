from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.static_routes.static_routes import (
def parse_af(self, item):
    match = re.search('(?:\\s*)(\\w+)(?:\\s*)(\\w+)', item, re.M)
    if match:
        return (match.group(1), match.group(2))