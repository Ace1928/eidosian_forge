import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_os_types(self):
    """
        List all registered os types (needed for snapshot creation)

        :rtype: ``list``
        """
    ostypes = self._sync_request('listOsTypes')
    return ostypes['ostype']