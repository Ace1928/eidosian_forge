from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient import exc
def raid_logical_disk_properties(self, driver_name, os_ironic_api_version=None, global_request_id=None):
    """Returns the RAID logical disk properties for the driver.

        :param driver_name: Name of the driver.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        :returns: A dictionary containing the properties that can be mentioned
            for RAID logical disks and a textual description for them. It
            returns an empty dictionary on error.
        """
    info = None
    try:
        info = self._list('/v1/drivers/%s/raid/logical_disk_properties' % driver_name, os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)[0]
    except IndexError:
        pass
    if info:
        return info.to_dict()
    return {}