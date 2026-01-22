from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def floating_ip_add(self, server, address, fixed_address=None):
    """Add a floating IP to a server

        :param server:
            The :class:`Server` (or its ID) to add an IP to.
        :param address:
            The FloatingIP or string floating address to add.
        :param fixed_address:
            The FixedIP the floatingIP should be associated with (optional)
        """
    url = '/servers'
    server = self.find(url, attr='name', value=server)
    address = address.ip if hasattr(address, 'ip') else address
    if fixed_address:
        if hasattr(fixed_address, 'ip'):
            fixed_address = fixed_address.ip
        body = {'address': address, 'fixed_address': fixed_address}
    else:
        body = {'address': address}
    return self._request('POST', '/%s/%s/action' % (url, server['id']), json={'addFloatingIp': body})