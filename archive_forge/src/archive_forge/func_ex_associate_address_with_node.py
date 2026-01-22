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
def ex_associate_address_with_node(self, node, elastic_ip, domain=None):
    """
        Associate an Elastic IP address with a particular node.

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      elastic_ip: Elastic IP instance
        :type       elastic_ip: :class:`ElasticIP`

        :param      domain: The domain where the IP resides (vpc only)
        :type       domain: ``str``

        :return:    A string representation of the association ID which is
                    required for VPC disassociation. EC2/standard
                    addresses return None
        :rtype:     ``None`` or ``str``
        """
    params = {'Action': 'AssociateAddress', 'InstanceId': node.id}
    if domain is not None and domain != 'vpc':
        raise AttributeError('Domain can only be set to vpc')
    if domain is None:
        params.update({'PublicIp': elastic_ip.ip})
    else:
        params.update({'AllocationId': elastic_ip.extra['allocation_id']})
    response = self.connection.request(self.path, params=params).object
    association_id = findtext(element=response, xpath='associationId', namespace=NAMESPACE)
    return association_id