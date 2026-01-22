from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_virtual_switch(self, data):
    match = re.search('^Virtual switch domain number : ([0-9]+)', data, re.M)
    if match:
        self.facts['virtual_switch'] = 'VSS'
        self.facts['virtual_switch_domain'] = match.group(1)
    match = re.findall('System\\".*?SN:\\s*([^\\s]+)', data, re.S)
    if match:
        self.facts['virtual_switch_serialnums'] = match