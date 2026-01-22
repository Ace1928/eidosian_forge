from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.six import raise_from
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def build_inventory_groups_location(self, group_name):
    """create group by attribute: location

        Args:
            str(group_name): Group name
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    if group_name not in self.inventory.groups:
        self.inventory.add_group(group_name)
    for instance_name in self.inventory.hosts:
        if 'ansible_lxd_location' in self.inventory.get_host(instance_name).get_vars():
            self.inventory.add_child(group_name, instance_name)