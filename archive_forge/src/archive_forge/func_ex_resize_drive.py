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
def ex_resize_drive(self, drive, size):
    """
        Resize a drive.

        :param drive: Drive to resize.

        :param size: New drive size in GBs.
        :type size: ``int``

        :return: Drive object which is being resized.
        :rtype: :class:`.CloudSigmaDrive`
        """
    path = '/drives/%s/action/' % drive.id
    data = {'name': drive.name, 'size': size * 1024 * 1024 * 1024, 'media': 'disk'}
    response = self._perform_action(path=path, action='resize', method='POST', data=data)
    drive = self._to_drive(data=response.object['objects'][0])
    return drive