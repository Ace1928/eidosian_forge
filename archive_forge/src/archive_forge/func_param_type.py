import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@classmethod
def param_type(cls):
    return PTYPE_IPV6