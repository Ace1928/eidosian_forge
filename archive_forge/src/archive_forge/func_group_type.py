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
def group_type(group_name):
    """create groups defined by lxd.yml or defaultvalues

            create groups defined by lxd.yml or defaultvalues
            supportetd:
                * 'location'
                * 'pattern'
                * 'network_range'
                * 'os'
                * 'release'
                * 'profile'
                * 'vlanid'
                * 'type'
                * 'project'

            Args:
                str(group_name): Group name
            Kwargs:
                None
            Raises:
                None
            Returns:
                None"""
    if self.groupby[group_name].get('type') == 'location':
        self.build_inventory_groups_location(group_name)
    elif self.groupby[group_name].get('type') == 'pattern':
        self.build_inventory_groups_pattern(group_name)
    elif self.groupby[group_name].get('type') == 'network_range':
        self.build_inventory_groups_network_range(group_name)
    elif self.groupby[group_name].get('type') == 'os':
        self.build_inventory_groups_os(group_name)
    elif self.groupby[group_name].get('type') == 'release':
        self.build_inventory_groups_release(group_name)
    elif self.groupby[group_name].get('type') == 'profile':
        self.build_inventory_groups_profile(group_name)
    elif self.groupby[group_name].get('type') == 'vlanid':
        self.build_inventory_groups_vlanid(group_name)
    elif self.groupby[group_name].get('type') == 'type':
        self.build_inventory_groups_type(group_name)
    elif self.groupby[group_name].get('type') == 'project':
        self.build_inventory_groups_project(group_name)
    else:
        raise AnsibleParserError('Unknown group type: {0}'.format(to_native(group_name)))