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
def ex_get_security_groups(self, group_ids=None, group_names=None, filters=None):
    """
        Returns a list of :class:`EC2SecurityGroup` objects for the
        current region.

        :param      group_ids: Returns only groups matching the provided
                               group IDs.
        :type       group_ids: ``list``

        :param      group_names: Returns only groups matching the provided
                                 group names.
        :type       group_ids: ``list``

        :param      filters: The filters so that the list returned includes
                             information for specific security groups only.
        :type       filters: ``dict``

        :rtype:     ``list`` of :class:`EC2SecurityGroup`
        """
    params = {'Action': 'DescribeSecurityGroups'}
    if group_ids:
        params.update(self._pathlist('GroupId', group_ids))
    if group_names:
        for name_idx, group_name in enumerate(group_names):
            name_idx += 1
            name_key = 'GroupName.%s' % name_idx
            params[name_key] = group_name
    if filters:
        params.update(self._build_filters(filters))
    response = self.connection.request(self.path, params=params)
    return self._to_security_groups(response.object)