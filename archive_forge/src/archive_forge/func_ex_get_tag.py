import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_get_tag(self, tag_id):
    """
        Retrieve a single tag.

        :param tag_id: ID of the tag to retrieve.
        :type tag_id: ``str``

        :rtype: ``list`` of :class:`.CloudSigmaTag` objects
        """
    action = '/tags/%s/' % tag_id
    response = self.connection.request(action=action, method='GET').object
    tag = self._to_tag(data=response)
    return tag