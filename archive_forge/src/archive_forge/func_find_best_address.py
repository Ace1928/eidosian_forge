import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def find_best_address(addresses, public=False, cloud_public=True):
    do_check = public == cloud_public
    if not addresses:
        return None
    if len(addresses) == 1:
        return addresses[0]
    if len(addresses) > 1 and do_check:
        for address in addresses:
            try:
                for count in utils.iterate_timeout(5, 'Timeout waiting for %s' % address, wait=0.1):
                    try:
                        for res in socket.getaddrinfo(address, 22, socket.AF_UNSPEC, socket.SOCK_STREAM, 0):
                            family, socktype, proto, _, sa = res
                            connect_socket = socket.socket(family, socktype, proto)
                            connect_socket.settimeout(1)
                            connect_socket.connect(sa)
                            return address
                    except socket.error:
                        continue
            except Exception:
                pass
    if do_check:
        log = _log.setup_logging('openstack')
        log.debug("The cloud returned multiple addresses %s:, and we could not connect to port 22 on either. That might be what you wanted, but we have no clue what's going on, so we picked the first one %s" % (addresses, addresses[0]))
    return addresses[0]