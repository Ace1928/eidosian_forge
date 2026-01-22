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
def ex_update_network_profile_of_node(self, node, network_profile):
    """
        Update the network profile of a node. This method can be used to
        attach or detach a NIC to a node.

        :param node: A node to attach the network interface to.
        :type node: :class:`Node`

        :param network_profile: The new network profile to update.
        :type network_profile: ``dict``
        """
    action = node.extra['id']
    location = node.extra['location']
    self.connection.request(action, method='PUT', params={'api-version': VM_API_VERSION}, data={'id': node.id, 'name': node.name, 'type': 'Microsoft.Compute/virtualMachines', 'location': location, 'properties': {'networkProfile': network_profile}})