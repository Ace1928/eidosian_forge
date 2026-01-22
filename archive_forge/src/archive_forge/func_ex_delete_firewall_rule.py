import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_firewall_rule(self, firewall_rule):
    """
        Remove a Firewall Rule.

        :param firewall_rule: Firewall rule which should be used
        :type  firewall_rule: :class:`CloudStackFirewallRule`

        :rtype: ``bool``
        """
    res = self._async_request(command='deleteFirewallRule', params={'id': firewall_rule.id}, method='GET')
    return res['success']