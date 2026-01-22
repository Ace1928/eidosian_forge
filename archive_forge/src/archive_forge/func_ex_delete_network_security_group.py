import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
def ex_delete_network_security_group(self, name, resource_group, location=None):
    """
        Update tags on any resource supporting tags.

        :param name: Name of the network security group to delete
        :type name: ``str``

        :param resource_group: The resource group to create the network
        security group in
        :type resource_group: ``str``

        :param location: The location at which to create the network security
        group (if None, use default location specified as 'region' in __init__)
        :type location: :class:`.NodeLocation`
        """
    if location is None:
        if self.default_location:
            location = self.default_location
        else:
            raise ValueError('location is required.')
    target = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups/%s' % (self.subscription_id, resource_group, name)
    data = {'location': location.id}
    self.connection.request(target, params={'api-version': NSG_API_VERSION}, data=data, method='DELETE')