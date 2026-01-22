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
def ex_associate_route_table(self, route_table, subnet):
    """
        Associates a route table with a subnet within a VPC.

        Note: A route table can be associated with multiple subnets.

        :param      route_table: The route table to associate.
        :type       route_table: :class:`.EC2RouteTable`

        :param      subnet: The subnet to associate with.
        :type       subnet: :class:`.EC2Subnet`

        :return:    Route table association ID.
        :rtype:     ``str``
        """
    params = {'Action': 'AssociateRouteTable', 'RouteTableId': route_table.id, 'SubnetId': subnet.id}
    result = self.connection.request(self.path, params=params).object
    association_id = findtext(element=result, xpath='associationId', namespace=NAMESPACE)
    return association_id