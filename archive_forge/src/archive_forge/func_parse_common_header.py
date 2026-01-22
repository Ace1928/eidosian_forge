import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
@classmethod
def parse_common_header(cls, buf):
    header_fields = struct.unpack_from(cls._HEADER_FMT, buf)
    return (list(header_fields), buf[cls.HEADER_SIZE:])