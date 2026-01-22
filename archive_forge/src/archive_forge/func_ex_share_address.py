import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
def ex_share_address(self, node, addresses):
    """Shares an IP with another node.This can be used to allow one Linode
         to begin serving requests should another become unresponsive.

        :param node: Node to share the IP addresses with
        :type node: :class:`Node`

        :keyword addresses: List of IP addresses to share
        :type address_type: `list` of :class: `LinodeIPAddress`

        :rtype: ``bool``
        """
    if not isinstance(node, Node):
        raise LinodeExceptionV4('Invalid node instance')
    if not all((isinstance(address, LinodeIPAddress) for address in addresses)):
        raise LinodeExceptionV4('Invalid address instance')
    attr = {'ips': [address.inet for address in addresses], 'linode_id': int(node.id)}
    response = self.connection.request('/v4/networking/ipv4/share', data=json.dumps(attr), method='POST')
    return response.status == httplib.OK