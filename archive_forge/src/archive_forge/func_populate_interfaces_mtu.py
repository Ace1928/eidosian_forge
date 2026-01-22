from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_interfaces_mtu(self, data):
    match = re.search('Jumbo frames are enabled', data, re.M)
    if match:
        mtu = 9000
    else:
        mtu = 1518
    self._mtu = mtu