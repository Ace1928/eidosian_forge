from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
def cidr_to_glob(cidr):
    """
    A function that accepts an IP subnet in a glob-style format and returns
    a list of CIDR subnets that exactly matches the specified glob.

    :param cidr: an IP object CIDR subnet.

    :return: a list of one or more IP addresses and subnets.
    """
    ip = IPNetwork(cidr)
    globs = iprange_to_globs(ip[0], ip[-1])
    if len(globs) != 1:
        raise AddrConversionError('bad CIDR to IP glob conversion!')
    return globs[0]