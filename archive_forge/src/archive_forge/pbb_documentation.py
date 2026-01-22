import struct
from os_ken.lib.packet import packet_base
I-TAG (IEEE 802.1ah-2008) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    pcp            Priority Code Point
    dei            Drop Eligible Indication
    uca            Use Customer Address
    sid            Service Instance ID
    ============== ====================
    