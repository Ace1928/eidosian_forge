import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_egress_firewall_rule(self, firewall_rule):
    """
        Remove a Firewall rule.

        :param egress_firewall_rule: Firewall rule which should be used
        :type  egress_firewall_rule: :class:`CloudStackEgressFirewallRule`

        :rtype: ``bool``
        """
    res = self._async_request(command='deleteEgressFirewallRule', params={'id': firewall_rule.id}, method='GET')
    return res['success']