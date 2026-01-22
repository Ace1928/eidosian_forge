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
def ex_replace_storage_volume_config(self, pool_id, type, name, definition):
    """
        Replace the storage volume information
        :param pool_id:
        :param type:
        :param name:
        :param definition
        """
    if not definition:
        raise LXDAPIException('Cannot create a storage volume without a definition')
    data = json.dumps(definition)
    response = self.connection.request('/{}/storage-pools/{}/volumes/{}/{}'.format(self.version, pool_id, type, name), method='PUT', data=data)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return self.ex_get_storage_pool_volume(pool_id=pool_id, type=type, name=name)