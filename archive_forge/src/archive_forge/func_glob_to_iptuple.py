from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
def glob_to_iptuple(ipglob):
    """
    A function that accepts a glob-style IP range and returns the component
    lower and upper bound IP address.

    :param ipglob: an IP address range in a glob-style format.

    :return: a tuple contain lower and upper bound IP objects.
    """
    if not valid_glob(ipglob):
        raise AddrFormatError('not a recognised IP glob range: %r!' % (ipglob,))
    start_tokens = []
    end_tokens = []
    for octet in ipglob.split('.'):
        if '-' in octet:
            tokens = octet.split('-')
            start_tokens.append(tokens[0])
            end_tokens.append(tokens[1])
        elif octet == '*':
            start_tokens.append('0')
            end_tokens.append('255')
        else:
            start_tokens.append(octet)
            end_tokens.append(octet)
    return (IPAddress('.'.join(start_tokens)), IPAddress('.'.join(end_tokens)))