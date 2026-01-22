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
def ex_allocate_address(self, domain='standard'):
    """
        Allocate a new Elastic IP address for EC2 classic or VPC

        :param      domain: The domain to allocate the new address in
                            (standard/vpc)
        :type       domain: ``str``

        :return:    Instance of ElasticIP
        :rtype:     :class:`ElasticIP`
        """
    params = {'Action': 'AllocateAddress'}
    if domain == 'vpc':
        params['Domain'] = domain
    response = self.connection.request(self.path, params=params).object
    return self._to_address(response, only_associated=False)