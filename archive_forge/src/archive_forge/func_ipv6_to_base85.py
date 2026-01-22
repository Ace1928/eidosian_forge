from netaddr.core import AddrFormatError
from netaddr.ip import IPAddress
def ipv6_to_base85(addr):
    """Convert a regular IPv6 address to base 85."""
    ip = IPAddress(addr)
    int_val = int(ip)
    remainder = []
    while int_val > 0:
        remainder.append(int_val % 85)
        int_val //= 85
    encoded = ''.join([BASE_85[w] for w in reversed(remainder)])
    leading_zeroes = (20 - len(encoded)) * '0'
    return leading_zeroes + encoded