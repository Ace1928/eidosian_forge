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
def build_inventory_groups_pattern(self, group_name):
    """create group by name pattern

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
    regex_pattern = self.groupby[group_name].get('attribute')
    for instance_name in self.inventory.hosts:
        result = re.search(regex_pattern, instance_name)
        if result:
            self.inventory.add_child(group_name, instance_name)