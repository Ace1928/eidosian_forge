import struct
from math import trunc
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils

    Internet Group Management Protocol(IGMP, RFC 3376)
    Membership Report Group Record message encoder/decoder class.

    http://www.ietf.org/rfc/rfc3376.txt

    This is used with os_ken.lib.packet.igmp.igmpv3_report.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    =============== ====================================================
    Attribute       Description
    =============== ====================================================
    type\_          a group record type for v3.
    aux_len         the length of the auxiliary data.
    num             a number of the multicast servers.
    address         a group address value.
    srcs            a list of IPv4 addresses of the multicast servers.
    aux             the auxiliary data.
    =============== ====================================================
    