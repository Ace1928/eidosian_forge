import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_network_acllist(self, name, vpc_id, description=None):
    """
        Create an ACL List for a network within a VPC.

        :param name: Name of the network ACL List
        :type  name: ``string``

        :param vpc_id: Id of the VPC associated with this network ACL List
        :type  vpc_id: ``string``

        :param description: Description of the network ACL List
        :type  description: ``string``

        :rtype: :class:`CloudStackNetworkACLList`
        """
    args = {'name': name, 'vpcid': vpc_id}
    if description:
        args['description'] = description
    result = self._sync_request(command='createNetworkACLList', params=args, method='GET')
    acl_list = CloudStackNetworkACLList(result['id'], name, vpc_id, self, description)
    return acl_list