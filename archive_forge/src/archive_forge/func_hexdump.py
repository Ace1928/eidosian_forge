import logging
from pyasn1 import __version__
from pyasn1 import error
from pyasn1.compat.octets import octs2ints
def hexdump(octets):
    return ' '.join(['%s%.2X' % (n % 16 == 0 and '\n%.5d: ' % n or '', x) for n, x in zip(range(len(octets)), octs2ints(octets))])