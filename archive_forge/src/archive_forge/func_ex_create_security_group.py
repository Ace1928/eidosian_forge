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
def ex_create_security_group(self, name, description, vpc_id=None):
    """
        Creates a new Security Group in EC2-Classic or a targeted VPC.

        :param      name:        The name of the security group to create.
                                 This must be unique.
        :type       name:        ``str``

        :param      description: Human readable description of a Security
                                 Group.
        :type       description: ``str``

        :param      vpc_id:      Optional identifier for VPC networks
        :type       vpc_id:      ``str``

        :rtype: ``dict``
        """
    params = {'Action': 'CreateSecurityGroup', 'GroupName': name, 'GroupDescription': description}
    if vpc_id is not None:
        params['VpcId'] = vpc_id
    response = self.connection.request(self.path, params=params).object
    group_id = findattr(element=response, xpath='groupId', namespace=NAMESPACE)
    return {'group_id': group_id}