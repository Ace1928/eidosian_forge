from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def zone_delete(self, zone_name):
    LOG.debug("Deleting DNS Zone '%s'" % zone_name)
    zone_to_be_deleted = self._get_zone(zone_name)
    if zone_to_be_deleted:
        zone_to_be_deleted.Delete_()