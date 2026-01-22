import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def cidr_abbrev_to_verbose(abbrev_cidr):
    """
    A function that converts abbreviated IPv4 CIDRs to their more verbose
    equivalent.

    :param abbrev_cidr: an abbreviated CIDR.

    Uses the old-style classful IP address rules to decide on a default
    subnet prefix if one is not explicitly provided.

    Only supports IPv4 addresses.

    Examples ::

        10                  - 10.0.0.0/8
        10/16               - 10.0.0.0/16
        128                 - 128.0.0.0/16
        128/8               - 128.0.0.0/8
        192.168             - 192.168.0.0/16

    :return: A verbose CIDR from an abbreviated CIDR or old-style classful         network address. The original value if it was not recognised as a         supported abbreviation.
    """

    def classful_prefix(octet):
        octet = int(octet)
        if not 0 <= octet <= 255:
            raise IndexError('Invalid octet: %r!' % octet)
        if 0 <= octet <= 127:
            return 8
        elif 128 <= octet <= 191:
            return 16
        elif 192 <= octet <= 223:
            return 24
        elif 224 <= octet <= 239:
            return 4
        return 32
    if isinstance(abbrev_cidr, str):
        if ':' in abbrev_cidr or abbrev_cidr == '':
            return abbrev_cidr
    try:
        i = int(abbrev_cidr)
        return '%s.0.0.0/%s' % (i, classful_prefix(i))
    except ValueError:
        if '/' in abbrev_cidr:
            part_addr, prefix = abbrev_cidr.split('/', 1)
            try:
                if not 0 <= int(prefix) <= 32:
                    raise ValueError('prefixlen in address %r out of range for IPv4!' % (abbrev_cidr,))
            except ValueError:
                return abbrev_cidr
        else:
            part_addr = abbrev_cidr
            prefix = None
        tokens = part_addr.split('.')
        if len(tokens) > 4:
            return abbrev_cidr
        for i in range(4 - len(tokens)):
            tokens.append('0')
        if prefix is None:
            try:
                prefix = classful_prefix(tokens[0])
            except ValueError:
                return abbrev_cidr
        return '%s/%s' % ('.'.join(tokens), prefix)
    except (TypeError, IndexError):
        return abbrev_cidr