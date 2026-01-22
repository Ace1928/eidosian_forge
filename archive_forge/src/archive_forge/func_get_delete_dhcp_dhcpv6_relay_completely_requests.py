from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_dhcp_dhcpv6_relay_completely_requests(self, have):
    """Get requests to delete all existing DHCP and DHCPv6 relay
        configurations in the chassis
        """
    requests = []
    for cfg in have:
        if cfg.get('ipv4'):
            requests.append(self.get_delete_all_dhcp_relay_intf_request(cfg['name']))
        if cfg.get('ipv6'):
            requests.append(self.get_delete_all_dhcpv6_relay_intf_request(cfg['name']))
    return requests