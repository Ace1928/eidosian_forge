import platform
import socket
import struct
from os_ken.lib import addrconv
def sa_in6(addr, port=0, flowinfo=0, scope_id=0):
    data = struct.pack('!HI16sI', port, flowinfo, addrconv.ipv6.text_to_bin(addr), scope_id)
    hdr = _hdr(_HDR_LEN + len(data), socket.AF_INET6)
    return hdr + data