import socket
import struct
def join_ipv4_segments(segments):
    """
    Helper method to join ip numeric segment pieces back into a full
    ip address.

    :param segments: IPv4 segments to join.
    :type segments: ``list`` or ``tuple``

    :return: IPv4 address.
    :rtype: ``str``
    """
    return '.'.join([str(s) for s in segments])