import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_port_forwarding_rule(self, node, rule):
    """
        Remove a Port forwarding rule.

        :param node: Node used in the rule
        :type  node: :class:`CloudStackNode`

        :param rule: Forwarding rule which should be used
        :type  rule: :class:`CloudStackPortForwardingRule`

        :rtype: ``bool``
        """
    node.extra['port_forwarding_rules'].remove(rule)
    node.public_ips.remove(rule.address.address)
    res = self._async_request(command='deletePortForwardingRule', params={'id': rule.id}, method='GET')
    return res['success']