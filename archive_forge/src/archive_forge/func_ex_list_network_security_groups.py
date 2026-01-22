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
def ex_list_network_security_groups(self, resource_group):
    """
        List network security groups.

        :param resource_group: List security groups in a specific resource
        group.
        :type resource_group: ``str``

        :return: A list of network security groups.
        :rtype: ``list`` of :class:`.AzureNetworkSecurityGroup`
        """
    action = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups' % (self.subscription_id, resource_group)
    r = self.connection.request(action, params={'api-version': NSG_API_VERSION})
    return [AzureNetworkSecurityGroup(net['id'], net['name'], net['location'], net['properties']) for net in r.object['value']]