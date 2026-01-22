from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
def valid_glob(ipglob):
    """
    :param ipglob: An IP address range in a glob-style format.

    :return: ``True`` if IP range glob is valid, ``False`` otherwise.
    """
    if not isinstance(ipglob, str):
        return False
    seen_hyphen = False
    seen_asterisk = False
    octets = ipglob.split('.')
    if len(octets) != 4:
        return False
    for octet in octets:
        if '-' in octet:
            if seen_hyphen:
                return False
            seen_hyphen = True
            if seen_asterisk:
                return False
            try:
                octet1, octet2 = [int(i) for i in octet.split('-')]
            except ValueError:
                return False
            if octet1 >= octet2:
                return False
            if not 0 <= octet1 <= 254:
                return False
            if not 1 <= octet2 <= 255:
                return False
        elif octet == '*':
            seen_asterisk = True
        else:
            if seen_hyphen is True:
                return False
            if seen_asterisk is True:
                return False
            try:
                if not 0 <= int(octet) <= 255:
                    return False
            except ValueError:
                return False
    return True