import array
import socket
import struct
from os_ken.lib import addrconv
def checksum_ip(ipvx, length, payload):
    """
    calculate checksum of IP pseudo header

    IPv4 pseudo header
    UDP RFC768
    TCP RFC793 3.1

     0      7 8     15 16    23 24    31
    +--------+--------+--------+--------+
    |          source address           |
    +--------+--------+--------+--------+
    |        destination address        |
    +--------+--------+--------+--------+
    |  zero  |protocol|    length       |
    +--------+--------+--------+--------+


    IPv6 pseudo header
    RFC2460 8.1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    +                                                               +
    |                                                               |
    +                         Source Address                        +
    |                                                               |
    +                                                               +
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    +                                                               +
    |                                                               |
    +                      Destination Address                      +
    |                                                               |
    +                                                               +
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                   Upper-Layer Packet Length                   |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                      zero                     |  Next Header  |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    """
    if ipvx.version == 4:
        header = struct.pack(_IPV4_PSEUDO_HEADER_PACK_STR, addrconv.ipv4.text_to_bin(ipvx.src), addrconv.ipv4.text_to_bin(ipvx.dst), ipvx.proto, length)
    elif ipvx.version == 6:
        header = struct.pack(_IPV6_PSEUDO_HEADER_PACK_STR, addrconv.ipv6.text_to_bin(ipvx.src), addrconv.ipv6.text_to_bin(ipvx.dst), length, ipvx.nxt)
    else:
        raise ValueError('Unknown IP version %d' % ipvx.version)
    buf = header + payload
    return checksum(buf)