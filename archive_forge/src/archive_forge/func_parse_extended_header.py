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
def parse_extended_header(cls, buf):
    ms_timestamp, = struct.unpack_from(cls._EXT_HEADER_FMT, buf)
    return ([ms_timestamp], buf[cls.EXT_HEADER_SIZE:])