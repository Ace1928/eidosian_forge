from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_structured_fan_info(self, data):
    objects = list()
    for key in ('fandetails', 'fandetails_3k'):
        if data.get(key):
            try:
                data = data[key]['TABLE_faninfo']['ROW_faninfo']
            except KeyError:
                pass
            else:
                objects = list(self.transform_iterable(data, self.FAN_MAP))
            break
    return objects