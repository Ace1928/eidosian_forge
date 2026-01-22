from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_stacks(self, data):
    match = re.findall('^Model [Nn]umber\\s+: (\\S+)', data, re.M)
    if match:
        self.facts['stacked_models'] = match
    match = re.findall('^System [Ss]erial [Nn]umber\\s+: (\\S+)', data, re.M)
    if match:
        self.facts['stacked_serialnums'] = match
    if 'stacked_models' in self.facts:
        self.facts['virtual_switch'] = 'STACK'