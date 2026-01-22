import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_disk_offerings(self):
    """
        Fetch a list of all available disk offerings.

        :rtype: ``list`` of :class:`CloudStackDiskOffering`
        """
    diskOfferings = []
    diskOfferResponse = self._sync_request(command='listDiskOfferings', method='GET')
    for diskOfferDict in diskOfferResponse.get('diskoffering', ()):
        diskOfferings.append(CloudStackDiskOffering(id=diskOfferDict['id'], name=diskOfferDict['name'], size=diskOfferDict['disksize'], customizable=diskOfferDict['iscustomized']))
    return diskOfferings