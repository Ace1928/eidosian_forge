from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from ansible.inventory.group import Group, InventoryObjectType
from ansible.parsing.utils.addresses import patterns
from ansible.utils.vars import combine_vars, get_unique_id
def populate_ancestors(self, additions=None):
    if additions is None:
        for group in self.groups:
            self.add_group(group)
    else:
        for group in additions:
            if group not in self.groups:
                self.groups.append(group)