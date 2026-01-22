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
def ex_describe_tags(self, resource):
    """
        Returns a dictionary of tags for a resource (e.g. Node or
        StorageVolume).

        :param  resource: The resource to be used
        :type   resource: any resource class, such as :class:`Node,`
                :class:`StorageVolume,` or :class:NodeImage`

        :return: A dictionary of Node tags
        :rtype: ``dict``
        """
    params = {'Action': 'DescribeTags'}
    filters = {'resource-id': resource.id}
    params.update(self._build_filters(filters))
    result = self.connection.request(self.path, params=params).object
    return self._get_resource_tags(result)