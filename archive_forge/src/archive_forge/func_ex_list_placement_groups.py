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
def ex_list_placement_groups(self, names=None):
    """
        A list of placement groups.

        :param names: Placement Group names
        :type names: ``list`` of ``str``

        :rtype: ``list`` of :class:`.EC2PlacementGroup`
        """
    names = names or []
    params = {'Action': 'DescribePlacementGroups'}
    for index, name in enumerate(names):
        params['GroupName.%s' % index + 1] = name
    response = self.connection.request(self.path, params=params).object
    return self._to_placement_groups(response)