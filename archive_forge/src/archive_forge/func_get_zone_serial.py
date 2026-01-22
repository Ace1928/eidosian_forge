from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def get_zone_serial(self, zone_name):
    if not self.zone_exists(zone_name):
        return None
    zone_soatype = self._dns_manager.MicrosoftDNS_SOAType(ContainerName=zone_name)
    if not zone_soatype:
        return None
    SOA = zone_soatype[0].SerialNumber
    return int(SOA)