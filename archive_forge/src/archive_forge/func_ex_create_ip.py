import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_create_ip(self, family, location, name):
    """
        Create either an ip_v4 ip or a ip_v6.

        :param family: Defines if the ip is v4 or v6 with int 4 or int 6.
        :type family: ``int``

        :param location: Defines which datacenter the created ip
                         responds with.
        :type location: :class:`.NodeLocation`

        :param name: Name of your Ip.
        :type name: ``str``

        :return: Ip
        :rtype: :class:`.GridscaleIp`
        """
    self.connection.async_request('objects/ips/', data={'name': name, 'family': family, 'location_uuid': location.id}, method='POST')
    return self._to_ip(self._get_resource('ips', self.connection.poll_response_initial.object['object_uuid']))