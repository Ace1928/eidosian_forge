import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_detach_floating_ip_from_node(self, node, ip):
    """
        Detach a floating IP from the given node

        Note: the 'node' object is not used in this method but it is added
        to the signature of ex_detach_floating_ip_from_node anyway so it
        conforms to the interface of the method of the same name for other
        drivers like for example OpenStack.

        :param      node: Node from which the IP should be detached
        :type       node: :class:`Node`

        :param      ip: Floating IP to detach
        :type       ip: :class:`DigitalOcean_v2_FloatingIpAddress`

        :rtype: ``bool``
        """
    data = {'type': 'unassign'}
    resp = self.connection.request('/v2/floating_ips/%s/actions' % ip.ip_address, data=json.dumps(data), method='POST')
    return resp.status == httplib.CREATED