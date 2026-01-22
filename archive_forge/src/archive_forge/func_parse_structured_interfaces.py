from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_structured_interfaces(self, data):
    objects = list()
    data = data['TABLE_interface']['ROW_interface']
    if isinstance(data, dict):
        objects.append(data['interface'])
    elif isinstance(data, list):
        for item in data:
            objects.append(item['interface'])
    return objects