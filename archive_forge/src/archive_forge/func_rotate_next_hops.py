from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
def rotate_next_hops(self, next_hops):
    """This method iterates through the list of
            next hops for a given destination network
            and converts it to a dictionary of dictionaries.
            Each dictionary has a primary key indicated by the
            tuple of `dest_vrf`, `forward_router_address` and
            `interface` and the value of this key is a dictionary
            that contains all the other attributes of the next hop.

        :rtype: A dict
        :returns: A next_hops list in a dictionary of dictionaries format
        """
    next_hops_dict = {}
    for entry in next_hops:
        entry = entry.copy()
        key_list = []
        for x in ['dest_vrf', 'forward_router_address', 'interface']:
            if entry.get(x):
                key_list.append(entry.pop(x))
        key = tuple(key_list)
        next_hops_dict[key] = entry
    return next_hops_dict