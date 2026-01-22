from paste.util import intset
import socket
def ip2int(addr, lookup=True):
    return _parseAddr(addr, lookup=lookup)[0]