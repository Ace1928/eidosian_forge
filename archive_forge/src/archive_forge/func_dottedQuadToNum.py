import re
import sys
from pprint import pprint
def dottedQuadToNum(ip):
    """
    Convert decimal dotted quad string to long integer
    
    >>> int(dottedQuadToNum('1 '))
    1
    >>> int(dottedQuadToNum(' 1.2'))
    16777218
    >>> int(dottedQuadToNum(' 1.2.3 '))
    16908291
    >>> int(dottedQuadToNum('1.2.3.4'))
    16909060
    >>> dottedQuadToNum('255.255.255.255')
    4294967295
    >>> dottedQuadToNum('255.255.255.256')
    Traceback (most recent call last):
    ValueError: Not a good dotted-quad IP: 255.255.255.256
    """
    import socket, struct
    try:
        return struct.unpack('!L', socket.inet_aton(ip.strip()))[0]
    except socket.error:
        raise ValueError('Not a good dotted-quad IP: %s' % ip)
    return