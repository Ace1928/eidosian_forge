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
def interface_selection(instance_name):
    """Select instance Interface for inventory

            Logic:
                - get preferred_interface & prefered_instance_network_family -> str(IP)
                - first Interface from: network_interfaces prefered_instance_network_family -> str(IP)

            Args:
                str(instance_name): name of instance
            Kwargs:
                None
            Raises:
                None
            Returns:
                dict(interface_name: ip)"""
    prefered_interface = self._get_data_entry('inventory/{0}/preferred_interface'.format(instance_name))
    prefered_instance_network_family = self.prefered_instance_network_family
    ip_address = ''
    if prefered_interface:
        interface = self._get_data_entry('inventory/{0}/network_interfaces/{1}'.format(instance_name, prefered_interface))
        for config in interface:
            if config['family'] == prefered_instance_network_family:
                ip_address = config['address']
                break
    else:
        interfaces = self._get_data_entry('inventory/{0}/network_interfaces'.format(instance_name))
        for interface in interfaces.values():
            for config in interface:
                if config['family'] == prefered_instance_network_family:
                    ip_address = config['address']
                    break
    return ip_address