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
def ex_list_storage_pools(self, detailed=True):
    """
        Returns a list of storage pools defined currently defined on the host

        Description: list of storage pools
        Authentication: trusted
        Operation: sync

        ":rtype: list of StoragePool items
        """
    response = self.connection.request('/%s/storage-pools' % self.version)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    pools = []
    for pool_item in response_dict['metadata']:
        pool_name = pool_item.split('/')[-1]
        if not detailed:
            pools.append(self._to_storage_pool({'name': pool_name, 'driver': None, 'used_by': None, 'config': None, 'managed': None}))
        else:
            pools.append(self.ex_get_storage_pool(id=pool_name))
    return pools