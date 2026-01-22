from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_vlans(self, data):
    objects = list()
    for line in data.splitlines():
        if line == '':
            continue
        if line[0].isdigit():
            vlan = line.split()[0]
            objects.append(vlan)
    return objects