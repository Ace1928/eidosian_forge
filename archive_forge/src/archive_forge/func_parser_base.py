import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@classmethod
def parser_base(cls, buf, recog):
    _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
    ptype, = struct.unpack_from('!H', buf, cls._MIN_LEN)
    cls_ = recog.get(ptype)
    info = cls_.parser(buf[cls._MIN_LEN:])
    msg = cls(flags, length, info)
    return msg