from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def populate_structured_neighbors_cdp(self, data):
    objects = dict()
    data = data['TABLE_cdp_neighbor_detail_info']['ROW_cdp_neighbor_detail_info']
    if isinstance(data, dict):
        data = [data]
    for item in data:
        if 'intf_id' in item:
            local_intf = item['intf_id']
        else:
            local_intf = item['interface']
        objects[local_intf] = list()
        nbor = dict()
        nbor['port'] = item['port_id']
        nbor['host'] = nbor['sysname'] = item['device_id']
        objects[local_intf].append(nbor)
    return objects