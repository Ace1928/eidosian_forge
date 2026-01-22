import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@staticmethod
def register_cfm_opcode(type_):

    def _register_cfm_opcode(cls):
        cfm._CFM_OPCODE[type_] = cls
        return cls
    return _register_cfm_opcode