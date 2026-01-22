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
def ex_resize_volume(self, volume, new_size, resource_group):
    """
        Resize a volume.

        :param volume: A volume to resize.
        :type volume: :class:`StorageVolume`

        :param new_size: The new size to resize the volume to in Gib.
        :type new_size: ``int``

        :param resource_group: The name of the resource group in which to
            create the volume.
        :type resource_group: ``str``

        """
    action = '/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/disks/{volume_name}'.format(subscription_id=self.subscription_id, resource_group=resource_group, volume_name=volume.name)
    data = {'location': volume.extra['location'], 'properties': {'diskSizeGB': new_size, 'creationData': volume.extra['properties']['creationData']}}
    response = self.connection.request(action, method='PUT', params={'api-version': DISK_API_VERSION}, data=data)
    return self._to_volume(response.object, name=volume.name, ex_resource_group=resource_group)