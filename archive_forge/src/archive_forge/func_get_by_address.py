from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
def get_by_address(self, address, fields=None, os_ironic_api_version=None, global_request_id=None):
    """Get a port group with the specified MAC address.

        :param address: The MAC address of a portgroup.
        :param fields: Optional, a list with a specified set of fields
                       of the resource to be returned. Can not be used
                       when 'detail' is set.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.

        :returns: a :class:`Portgroup` object.

        """
    path = '?address=%s' % address
    if fields is not None:
        path += '&fields=' + ','.join(fields)
    else:
        path = 'detail' + path
    portgroups = self._list(self._path(path), 'portgroups', os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)
    if len(portgroups) == 1:
        return portgroups[0]
    else:
        raise exc.NotFound()