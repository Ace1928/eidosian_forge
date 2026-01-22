import socket
import struct
def is_valid_ip_address(address, family=socket.AF_INET):
    """
    Check if the provided address is valid IPv4 or IPv6 address.

    :param address: IPv4 or IPv6 address to check.
    :type address: ``str``

    :param family: Address family (socket.AF_INTET / socket.AF_INET6).
    :type family: ``int``

    :return: ``bool`` True if the provided address is valid.
    """
    try:
        socket.inet_pton(family, address)
    except OSError:
        return False
    return True