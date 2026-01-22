from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.zonomi import ZonomiResponse, ZonomiException, ZonomiConnection
def ex_convert_to_master(self, zone):
    """
        Convert existent zone to master.

        :param zone: Zone to convert.
        :type  zone: :class:`Zone`

        :rtype: Bool
        """
    action = '/app/dns/converttomaster.jsp?'
    params = {'name': zone.domain}
    try:
        self.connection.request(action=action, params=params)
    except ZonomiException as e:
        if 'ERROR: Could not find' in e.message:
            raise ZoneDoesNotExistError(zone_id=zone.id, driver=self, value=e.message)
    return True