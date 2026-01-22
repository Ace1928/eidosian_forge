from . import exceptions
from . import misc
from . import normalizers
def valid_ipv4_host_address(host):
    """Determine if the given host is a valid IPv4 address."""
    return all([0 <= int(byte, base=10) <= 255 for byte in host.split('.')])