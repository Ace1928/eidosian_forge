from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def vlan_list_to_range(cmd):
    """
    Converts a comma separated list of vlan IDs
    into ranges.
    """
    ranges = []
    for v in get_ranges(cmd):
        ranges.append('-'.join(map(str, (v[0], v[-1])[:len(v)])))
    return ','.join(ranges)