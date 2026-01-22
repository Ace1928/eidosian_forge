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
def ex_release_address(self, elastic_ip, domain=None):
    """
        Releases an Elastic IP address using the IP (EC2-Classic) or
        using the allocation ID (VPC).

        :param      elastic_ip: Elastic IP instance
        :type       elastic_ip: :class:`ElasticIP`

        :param      domain: The domain where the IP resides (vpc only)
        :type       domain: ``str``

        :return:    True on success, False otherwise.
        :rtype:     ``bool``
        """
    params = {'Action': 'ReleaseAddress'}
    if domain is not None and domain != 'vpc':
        raise AttributeError('Domain can only be set to vpc')
    if domain is None:
        params['PublicIp'] = elastic_ip.ip
    else:
        params['AllocationId'] = elastic_ip.extra['allocation_id']
    response = self.connection.request(self.path, params=params).object
    return self._get_boolean(response)