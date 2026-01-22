from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def zone_list(self):
    """Returns the current list of DNS Zones.

        """
    zones = self._dns_manager.MicrosoftDNS_Zone()
    return [x.Name for x in zones]