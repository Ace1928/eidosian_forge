from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_memtotal_mb(self, data):
    match = re.search('(\\S+)K(\\s+|)total', data, re.M)
    if match:
        memtotal = match.group(1)
        return int(memtotal) / 1024