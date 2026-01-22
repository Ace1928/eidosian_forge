from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient import exc
def vendor_passthru(self, driver_name, method, args=None, http_method=None, os_ironic_api_version=None, global_request_id=None):
    """Issue requests for vendor-specific actions on a given driver.

        :param driver_name: The name of the driver.
        :param method: Name of the vendor method.
        :param args: Optional. The arguments to be passed to the method.
        :param http_method: The HTTP method to use on the request.
                            Defaults to POST.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        """
    if args is None:
        args = {}
    if http_method is None:
        http_method = 'POST'
    http_method = http_method.upper()
    header_values = {'os_ironic_api_version': os_ironic_api_version, 'global_request_id': global_request_id}
    path = '%s/vendor_passthru/%s' % (driver_name, method)
    if http_method in ('POST', 'PUT', 'PATCH'):
        return self.update(path, args, http_method=http_method, **header_values)
    elif http_method == 'DELETE':
        return self.delete(path, **header_values)
    elif http_method == 'GET':
        return self.get(path, **header_values)
    else:
        raise exc.InvalidAttribute(_('Unknown HTTP method: %s') % http_method)