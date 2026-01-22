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
def ex_get_api_endpoints(self):
    """
        Description: List of supported APIs
        Authentication: guest
        Operation: sync
        Return: list of supported API endpoint URLs

        """
    response = self.connection.request('/')
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return response_dict['metadata']