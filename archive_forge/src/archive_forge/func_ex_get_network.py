import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def ex_get_network(self, name):
    """
        Returns the LXD network with the given name.
        Implements GET /1.0/networks/<name>

        Authentication: trusted
        Operation: sync

        :param name: The name of the network to return
        :type  name: str

        :rtype: LXDNetwork
        """
    req = '/{}/networks/{}'.format(self.version, name)
    response = self.connection.request(req)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return LXDNetwork.build_from_response(response_dict['metadata'])