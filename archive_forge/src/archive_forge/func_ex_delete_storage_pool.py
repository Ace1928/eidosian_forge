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
def ex_delete_storage_pool(self, id):
    """Delete the storage pool.

        Implements DELETE /1.0/storage-pools/<self.name>

        Deleting a storage pool may fail if it is being used.  See the LXD
        documentation for further details.

        :raises: :class:`LXDAPIException` if the storage pool can't be deleted.
        """
    req = '/{}/storage-pools/{}'.format(self.version, id)
    response = self.connection.request(req, method='DELETE')
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)