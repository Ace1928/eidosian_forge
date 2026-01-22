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
def ex_clone_drive(self, drive, name=None, ex_avoid=None):
    """
        Clone a library or a standard drive.

        :param drive: Drive to clone.
        :type drive: :class:`libcloud.compute.base.NodeImage` or
                     :class:`.CloudSigmaDrive`

        :param name: Optional name for the cloned drive.
        :type name: ``str``

        :param ex_avoid: A list of other drive uuids to avoid when
                         creating this drive. If provided, drive will
                         attempt to be created on a different
                         physical infrastructure from other drives
                         specified using this argument. (optional)
        :type ex_avoid: ``list``

        :return: New cloned drive.
        :rtype: :class:`.CloudSigmaDrive`
        """
    params = {}
    data = {}
    if ex_avoid:
        params['avoid'] = ','.join(ex_avoid)
    if name:
        data['name'] = name
    path = '/drives/%s/action/' % drive.id
    response = self._perform_action(path=path, action='clone', params=params, data=data, method='POST')
    drive = self._to_drive(data=response.object['objects'][0])
    return drive