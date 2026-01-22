import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify

    ICMPv6 sub encoder/decoder class for MLD v2 Lister Report Group
    Record messages. (RFC 3810)

    This is used with os_ken.lib.packet.icmpv6.mldv2_report.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    =============== ====================================================
    Attribute       Description
    =============== ====================================================
    type\_          a group record type for v3.
    aux_len         the length of the auxiliary data in 32-bit words.
    num             a number of the multicast servers.
    address         a group address value.
    srcs            a list of IPv6 addresses of the multicast servers.
    aux             the auxiliary data.
    =============== ====================================================
    