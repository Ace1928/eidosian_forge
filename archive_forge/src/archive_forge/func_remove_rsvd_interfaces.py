from __future__ import absolute_import, division, print_function
import socket
from functools import total_ordering
from itertools import count, groupby
from ansible.module_utils.six import iteritems
def remove_rsvd_interfaces(interfaces):
    """Exclude reserved interfaces from user management"""
    if not interfaces:
        return []
    return [i for i in interfaces if get_interface_type(i['name']) != 'management']