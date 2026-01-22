import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_network_acl(self):
    """
        Lists all network ACL items

        :rtype: ``list`` of :class:`CloudStackNetworkACL`
        """
    acls = []
    result = self._sync_request(command='listNetworkACLs', method='GET')
    if not result:
        return acls
    for acl in result['networkacl']:
        acls.append(CloudStackNetworkACL(acl['id'], acl['protocol'], acl['aclid'], acl['action'], acl['cidrlist'], acl.get('startport', []), acl.get('endport', []), acl['traffictype']))
    return acls