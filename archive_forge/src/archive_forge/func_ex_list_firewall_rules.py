import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_firewall_rules(self):
    """
        Lists all Firewall Rules

        :rtype: ``list`` of :class:`CloudStackFirewallRule`
        """
    rules = []
    result = self._sync_request(command='listFirewallRules', method='GET')
    if result != {}:
        public_ips = self.ex_list_public_ips()
        for rule in result['firewallrule']:
            addr = [a for a in public_ips if a.address == rule['ipaddress']]
            rules.append(CloudStackFirewallRule(rule['id'], addr[0], rule['cidrlist'], rule['protocol'], rule.get('icmpcode'), rule.get('icmptype'), rule.get('startport'), rule.get('endport')))
    return rules