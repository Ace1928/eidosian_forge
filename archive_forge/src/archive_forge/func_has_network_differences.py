from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def has_network_differences(self, container):
    """
        Check if the container is connected to requested networks with expected options: links, aliases, ipv4, ipv6
        """
    different = False
    differences = []
    if not self.module.params['networks']:
        return (different, differences)
    if not container.container.get('NetworkSettings'):
        self.fail('has_missing_networks: Error parsing container properties. NetworkSettings missing.')
    connected_networks = container.container['NetworkSettings']['Networks']
    for network in self.module.params['networks']:
        network_info = connected_networks.get(network['name'])
        if network_info is None:
            different = True
            differences.append(dict(parameter=network, container=None))
        else:
            diff = False
            network_info_ipam = network_info.get('IPAMConfig') or {}
            if network.get('ipv4_address') and network['ipv4_address'] != network_info_ipam.get('IPv4Address'):
                diff = True
            if network.get('ipv6_address') and network['ipv6_address'] != network_info_ipam.get('IPv6Address'):
                diff = True
            if network.get('aliases'):
                if not compare_generic(network['aliases'], network_info.get('Aliases'), 'allow_more_present', 'set'):
                    diff = True
            if network.get('links'):
                expected_links = []
                for link, alias in network['links']:
                    expected_links.append('%s:%s' % (link, alias))
                if not compare_generic(expected_links, network_info.get('Links'), 'allow_more_present', 'set'):
                    diff = True
            if network.get('mac_address') and network['mac_address'] != network_info.get('MacAddress'):
                diff = True
            if diff:
                different = True
                differences.append(dict(parameter=network, container=dict(name=network['name'], ipv4_address=network_info_ipam.get('IPv4Address'), ipv6_address=network_info_ipam.get('IPv6Address'), aliases=network_info.get('Aliases'), links=network_info.get('Links'), mac_address=network_info.get('MacAddress'))))
    return (different, differences)