import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_routers(self, vpc_id=None):
    """
        List routers

        :rtype ``list`` of :class:`CloudStackRouter`
        """
    args = {}
    if vpc_id is not None:
        args['vpcid'] = vpc_id
    res = self._sync_request(command='listRouters', params=args, method='GET')
    rts = res.get('router', [])
    routers = []
    for router in rts:
        routers.append(CloudStackRouter(router['id'], router['name'], router['state'], router['publicip'], router['vpcid'], self))
    return routers