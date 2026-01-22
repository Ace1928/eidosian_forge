import array
import socket
import struct
from os_ken.lib import addrconv
def carry_around_add(a, b):
    c = a + b
    return (c & 65535) + (c >> 16)