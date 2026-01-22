import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_create_floating_ip(self, location):
    """
        Create new floating IP reserved to a region.

        The newly created floating IP will not be associated to a Droplet.

        See https://developers.digitalocean.com/documentation/v2/#floating-ips

        :param location: Which data center to create the floating IP in.
        :type location: :class:`.NodeLocation`

        :rtype: :class:`DigitalOcean_v2_FloatingIpAddress`
        """
    attr = {'region': location.id}
    resp = self.connection.request('/v2/floating_ips', data=json.dumps(attr), method='POST')
    return self._to_floating_ip(resp.object['floating_ip'])