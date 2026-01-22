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
def ex_list_image_versions(self, sku):
    """
        List node image versions in a sku.

        :param sku: The complete resource path to a sku (as returned by
        `ex_list_skus`)
        :type publisher: ``str``

        :return: A list of tuples in the form
        ("version id", "version name")
        :rtype: ``list``
        """
    action = '%s/versions' % sku
    r = self.connection.request(action, params={'api-version': IMAGES_API_VERSION})
    return [(img['id'], img['name']) for img in r.object]