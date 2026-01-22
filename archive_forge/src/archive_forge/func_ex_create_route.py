import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def ex_create_route(self, route_table, cidr, internet_gateway=None, node=None, network_interface=None, vpc_peering_connection=None):
    """
        Creates a route entry in the route table.

        :param      route_table: The route table to create the route in.
        :type       route_table: :class:`.EC2RouteTable`

        :param      cidr: The CIDR block used for the destination match.
        :type       cidr: ``str``

        :param      internet_gateway: The Internet gateway to route
                                      traffic through.
        :type       internet_gateway: :class:`.VPCInternetGateway`

        :param      node: The NAT instance to route traffic through.
        :type       node: :class:`Node`

        :param      network_interface: The network interface of the node
                                       to route traffic through.
        :type       network_interface: :class:`.EC2NetworkInterface`

        :param      vpc_peering_connection: The VPC peering connection.
        :type       vpc_peering_connection: :class:`.VPCPeeringConnection`

        :rtype:     ``bool``

        Note: You must specify one of the following: internet_gateway,
              node, network_interface, vpc_peering_connection.
        """
    params = {'Action': 'CreateRoute', 'RouteTableId': route_table.id, 'DestinationCidrBlock': cidr}
    if internet_gateway:
        params['GatewayId'] = internet_gateway.id
    if node:
        params['InstanceId'] = node.id
    if network_interface:
        params['NetworkInterfaceId'] = network_interface.id
    if vpc_peering_connection:
        params['VpcPeeringConnectionId'] = vpc_peering_connection.id
    res = self.connection.request(self.path, params=params).object
    return self._get_boolean(res)