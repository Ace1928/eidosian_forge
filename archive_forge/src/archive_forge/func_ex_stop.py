import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_stop(self, node):
    """
        Stops/Suspends a running virtual machine

        :param node: Node to stop.
        :type node: :class:`CloudStackNode`

        :rtype: ``str``
        """
    res = self._async_request(command='stopVirtualMachine', params={'id': node.id}, method='GET')
    return res['virtualmachine']['state']