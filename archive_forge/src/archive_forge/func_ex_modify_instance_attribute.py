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
def ex_modify_instance_attribute(self, node, disable_api_termination=None, ebs_optimized=None, group_id=None, source_dest_check=None, user_data=None, instance_type=None, attributes=None):
    """
        Modifies node attributes.
        Ouscale supports the following attributes:
        'DisableApiTermination.Value', 'EbsOptimized', 'GroupId.n',
        'SourceDestCheck.Value', 'UserData.Value',
        'InstanceType.Value'

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      attributes: A dictionary with node attributes
        :type       attributes: ``dict``

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
    attributes = attributes or {}
    if disable_api_termination is not None:
        attributes['DisableApiTermination.Value'] = disable_api_termination
    if ebs_optimized is not None:
        attributes['EbsOptimized'] = ebs_optimized
    if group_id is not None:
        attributes['GroupId.n'] = group_id
    if source_dest_check is not None:
        attributes['SourceDestCheck.Value'] = source_dest_check
    if user_data is not None:
        attributes['UserData.Value'] = user_data
    if instance_type is not None:
        attributes['InstanceType.Value'] = instance_type
    return super().ex_modify_instance_attribute(node, attributes)