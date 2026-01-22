import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_replace_network_acllist(self, acl_id, network_id):
    """
        Create an ACL List for a network within a VPC.Replaces ACL associated
        with a Network or private gateway

        :param acl_id: the ID of the network ACL
        :type  acl_id: ``string``

        :param network_id: the ID of the network
        :type  network_id: ``string``

        :rtype: :class:`CloudStackNetworkACLList`
        """
    args = {'aclid': acl_id, 'networkid': network_id}
    self._async_request(command='replaceNetworkACLList', params=args, method='GET')
    return True