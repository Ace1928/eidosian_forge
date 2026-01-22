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
def get_prefered_instance_network_interface(self, instance_name):
    """Helper to get the preferred interface of thr instance

        Helper to get the preferred interface provide by neme pattern from 'prefered_instance_network_interface'.

        Args:
            str(instance_name): name of instance
        Kwargs:
            None
        Raises:
            None
        Returns:
            str(prefered_interface): None or interface name"""
    instance_network_interfaces = self._get_data_entry('inventory/{0}/network_interfaces'.format(instance_name))
    prefered_interface = None
    if instance_network_interfaces:
        net_generator = [interface for interface in instance_network_interfaces if interface.startswith(self.prefered_instance_network_interface)]
        selected_interfaces = []
        for interface in net_generator:
            selected_interfaces.append(interface)
        if len(selected_interfaces) > 0:
            prefered_interface = sorted(selected_interfaces)[0]
    return prefered_interface