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
def ex_list_skus(self, offer):
    """
        List node image skus in an offer.

        :param offer: The complete resource path to an offer (as returned by
        `ex_list_offers`)
        :type offer: ``str``

        :return: A list of tuples in the form
        ("sku id", "sku name")
        :rtype: ``list``
        """
    action = '%s/skus' % offer
    r = self.connection.request(action, params={'api-version': IMAGES_API_VERSION})
    return [(sku['id'], sku['name']) for sku in r.object]