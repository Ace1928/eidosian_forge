import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_vpc(self, vpc):
    """

        Deletes a VPC, only available in advanced zones.

        :param  vpc: The VPC
        :type   vpc: :class: 'CloudStackVPC'

        :rtype: ``bool``

        """
    args = {'id': vpc.id}
    self._async_request(command='deleteVPC', params=args, method='GET')
    return True