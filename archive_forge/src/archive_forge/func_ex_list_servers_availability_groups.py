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
def ex_list_servers_availability_groups(self):
    """
        Return which running servers share the same physical compute host.

        :return: A list of server UUIDs which share the same physical compute
                 host. Servers which share the same host will be stored under
                 the same list index.
        :rtype: ``list`` of ``list``
        """
    action = '/servers/availability_groups/'
    response = self.connection.request(action=action, method='GET')
    return response.object