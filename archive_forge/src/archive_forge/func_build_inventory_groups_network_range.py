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
def build_inventory_groups_network_range(self, group_name):
    """check if IP is in network-class

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
    try:
        network = ipaddress.ip_network(to_text(self.groupby[group_name].get('attribute')))
    except ValueError as err:
        raise AnsibleParserError('Error while parsing network range {0}: {1}'.format(self.groupby[group_name].get('attribute'), to_native(err)))
    for instance_name in self.inventory.hosts:
        if self.data['inventory'][instance_name].get('network_interfaces') is not None:
            for interface in self.data['inventory'][instance_name].get('network_interfaces'):
                for interface_family in self.data['inventory'][instance_name].get('network_interfaces')[interface]:
                    try:
                        address = ipaddress.ip_address(to_text(interface_family['address']))
                        if address.version == network.version and address in network:
                            self.inventory.add_child(group_name, instance_name)
                    except ValueError:
                        pass