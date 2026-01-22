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
def extract_network_information_from_instance_config(self, instance_name):
    """Returns the network interface configuration

        Returns the network ipv4 and ipv6 config of the instance without local-link

        Args:
            str(instance_name): Name oft he instance
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(network_configuration): network config"""
    instance_network_interfaces = self._get_data_entry('instances/{0}/state/metadata/network'.format(instance_name))
    network_configuration = None
    if instance_network_interfaces:
        network_configuration = {}
        gen_interface_names = [interface_name for interface_name in instance_network_interfaces if interface_name != 'lo']
        for interface_name in gen_interface_names:
            gen_address = [address for address in instance_network_interfaces[interface_name]['addresses'] if address.get('scope') != 'link']
            network_configuration[interface_name] = []
            for address in gen_address:
                address_set = {}
                address_set['family'] = address.get('family')
                address_set['address'] = address.get('address')
                address_set['netmask'] = address.get('netmask')
                address_set['combined'] = address.get('address') + '/' + address.get('netmask')
                network_configuration[interface_name].append(address_set)
    return network_configuration