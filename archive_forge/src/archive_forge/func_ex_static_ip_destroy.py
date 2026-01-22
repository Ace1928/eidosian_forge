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
def ex_static_ip_destroy(self, ip_address):
    """
        Destroy a static IP address.

        :param      ip_address: IP address which should be used
        :type       ip_address: ``str``

        :rtype: ``bool``
        """
    response = self.connection.request(action='/resources/ip/%s/destroy' % ip_address, method='GET')
    return response.status == 204