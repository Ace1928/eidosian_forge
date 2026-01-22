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
def ex_allocate_private_address(self, node, address_type='ipv4'):
    """Allocates a private IPv4 address to node.Only ipv4 is currently supported

        :param node: Node to attach the IP address
        :type node: :class:`Node`

        :keyword address_type: Type of IP address
        :type address_type: `str`

        :return: The newly created LinodeIPAddress
        :rtype: :class:`LinodeIPAddress`
        """
    if not isinstance(node, Node):
        raise LinodeExceptionV4('Invalid node instance')
    if address_type != 'ipv4':
        raise LinodeExceptionV4('Address type not supported')
    if len(node.private_ips) >= 1:
        raise LinodeExceptionV4('Nodes can have up to one private IP')
    attr = {'public': False, 'type': address_type}
    response = self.connection.request('/v4/linode/instances/%s/ips' % node.id, data=json.dumps(attr), method='POST').object
    return self._to_address(response)