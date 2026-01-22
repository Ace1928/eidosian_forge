import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@classmethod
def cause_code(cls):
    return CCODE_PROTOCOL_VIOLATION