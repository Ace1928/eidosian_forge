from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.zonomi import ZonomiResponse, ZonomiException, ZonomiConnection

        Convert existent zone to master.

        :param zone: Zone to convert.
        :type  zone: :class:`Zone`

        :rtype: Bool
        