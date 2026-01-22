from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def has_extra_networks(self, container):
    """
        Check if the container is connected to non-requested networks
        """
    extra_networks = []
    extra = False
    if not container.container.get('NetworkSettings'):
        self.fail('has_extra_networks: Error parsing container properties. NetworkSettings missing.')
    connected_networks = container.container['NetworkSettings'].get('Networks')
    if connected_networks:
        for network, network_config in connected_networks.items():
            keep = False
            if self.module.params['networks']:
                for expected_network in self.module.params['networks']:
                    if expected_network['name'] == network:
                        keep = True
            if not keep:
                extra = True
                extra_networks.append(dict(name=network, id=network_config['NetworkID']))
    return (extra, extra_networks)