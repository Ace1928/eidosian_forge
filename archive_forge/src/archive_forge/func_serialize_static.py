import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
@staticmethod
def serialize_static(vrrp_, prev):
    if isinstance(prev, ipv4.ipv4):
        assert type(vrrp_.ip_addresses[0]) == str
        conv = addrconv.ipv4.text_to_bin
        ip_address_pack_raw = vrrpv3._IPV4_ADDRESS_PACK_STR_RAW
    elif isinstance(prev, ipv6.ipv6):
        assert type(vrrp_.ip_addresses[0]) == str
        conv = addrconv.ipv6.text_to_bin
        ip_address_pack_raw = vrrpv3._IPV6_ADDRESS_PACK_STR_RAW
    else:
        raise ValueError('Unkown network layer %s' % type(prev))
    ip_addresses_pack_str = '!' + ip_address_pack_raw * vrrp_.count_ip
    ip_addresses_len = struct.calcsize(ip_addresses_pack_str)
    vrrp_len = vrrpv3._MIN_LEN + ip_addresses_len
    checksum = False
    if vrrp_.checksum is None:
        checksum = True
        vrrp_.checksum = 0
    buf = bytearray(vrrp_len)
    assert vrrp_.max_adver_int <= VRRP_V3_MAX_ADVER_INT_MASK
    struct.pack_into(vrrpv3._PACK_STR, buf, 0, vrrp_to_version_type(vrrp_.version, vrrp_.type), vrrp_.vrid, vrrp_.priority, vrrp_.count_ip, vrrp_.max_adver_int, vrrp_.checksum)
    struct.pack_into(ip_addresses_pack_str, buf, vrrpv3._MIN_LEN, *[conv(x) for x in vrrp_.ip_addresses])
    if checksum:
        vrrp_.checksum = packet_utils.checksum_ip(prev, len(buf), buf)
        struct.pack_into(vrrpv3._CHECKSUM_PACK_STR, buf, vrrpv3._CHECKSUM_OFFSET, vrrp_.checksum)
    return buf