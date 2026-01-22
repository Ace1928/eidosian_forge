import os
import traceback
from twisted.application import internet, service
from twisted.names import authority, dns, secondary, server
from twisted.python import usage
def opt_secondary(self, ip_domain):
    """Act as secondary for the specified domain, performing
        zone transfers from the specified IP (IP/domain)
        """
    args = ip_domain.split('/', 1)
    if len(args) != 2:
        raise usage.UsageError('Argument must be of the form IP[:port]/domain')
    address = args[0].split(':')
    if len(address) == 1:
        address = (address[0], dns.PORT)
    else:
        try:
            port = int(address[1])
        except ValueError:
            raise usage.UsageError(f'Specify an integer port number, not {address[1]!r}')
        address = (address[0], port)
    self.secondaries.append((address, [args[1]]))