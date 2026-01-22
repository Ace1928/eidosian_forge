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
def ex_create_internet_gateway(self, name=None):
    """
        Delete a VPC Internet gateway

        :rtype:     ``bool``
        """
    params = {'Action': 'CreateInternetGateway'}
    resp = self.connection.request(self.path, params=params).object
    element = resp.findall(fixxpath(xpath='internetGateway', namespace=NAMESPACE))
    gateway = self._to_internet_gateway(element[0], name)
    if name and self.ex_create_tags(gateway, {'Name': name}):
        gateway.extra['tags']['Name'] = name
    return gateway