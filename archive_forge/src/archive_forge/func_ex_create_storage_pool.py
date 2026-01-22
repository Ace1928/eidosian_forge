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
def ex_create_storage_pool(self, definition):
    """
        Create a storage_pool from definition.

        Implements POST /1.0/storage-pools

        The `definition` parameter defines
        what the storage pool will be.  An
        example config for the zfs driver is:

                   {
                       "config": {
                           "size": "10GB"
                       },
                       "driver": "zfs",
                       "name": "pool1"
                   }

        Note that **all** fields in the `definition` parameter are strings.
        Note that size has to be at least 64MB in order to create the pool

        For further details on the storage pool types see:
        https://lxd.readthedocs.io/en/latest/storage/

        The function returns the a `StoragePool` instance, if it is
        successfully created, otherwise an LXDAPIException is raised.

        :param definition: the fields to pass to the LXD API endpoint
        :type definition: dict

        :returns: a storage pool if successful,
        raises NotFound if not found
        :rtype: :class:`StoragePool`

        :raises: :class:`LXDAPIExtensionNotAvailable`
        if the 'storage' api extension is missing.
        :raises: :class:`LXDAPIException`
        if the storage pool couldn't be created.
        """
    if not definition:
        raise LXDAPIException('Cannot create a storage pool  without a definition')
    data = json.dumps(definition)
    response = self.connection.request('/%s/storage-pools' % self.version, method='POST', data=data)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return self.ex_get_storage_pool(id=definition['name'])