import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_vpcs(self, project=None):
    """
        List the available VPCs

        :keyword    project: Optional project under which VPCs are present.
        :type       project: :class:`.CloudStackProject`

        :rtype ``list`` of :class:`CloudStackVPC`
        """
    args = {}
    if project is not None:
        args['projectid'] = project.id
    res = self._sync_request(command='listVPCs', params=args, method='GET')
    vpcs = res.get('vpc', [])
    networks = []
    for vpc in vpcs:
        networks.append(CloudStackVPC(vpc['name'], vpc['vpcofferingid'], vpc['id'], vpc['cidr'], self, vpc['zoneid'], vpc['displaytext']))
    return networks