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
def create_packet(self, primary_ip_address, vlan_id=None):
    """Prepare a VRRP packet.

        Returns a newly created os_ken.lib.packet.packet.Packet object
        with appropriate protocol header objects added by add_protocol().
        It's caller's responsibility to serialize().
        The serialized packet would looks like the ones described in
        the following sections.

        * RFC 3768 5.1. VRRP Packet Format
        * RFC 5798 5.1. VRRP Packet Format

        ================== ====================
        Argument           Description
        ================== ====================
        primary_ip_address Source IP address
        vlan_id            VLAN ID.  None for no VLAN.
        ================== ====================
        """
    if self.is_ipv6:
        traffic_class = 192
        flow_label = 0
        payload_length = ipv6.ipv6._MIN_LEN + len(self)
        e = ethernet.ethernet(VRRP_IPV6_DST_MAC_ADDRESS, vrrp_ipv6_src_mac_address(self.vrid), ether.ETH_TYPE_IPV6)
        ip = ipv6.ipv6(6, traffic_class, flow_label, payload_length, inet.IPPROTO_VRRP, VRRP_IPV6_HOP_LIMIT, primary_ip_address, VRRP_IPV6_DST_ADDRESS)
    else:
        header_length = ipv4.ipv4._MIN_LEN // 4
        total_length = 0
        tos = 192
        identification = self.get_identification()
        e = ethernet.ethernet(VRRP_IPV4_DST_MAC_ADDRESS, vrrp_ipv4_src_mac_address(self.vrid), ether.ETH_TYPE_IP)
        ip = ipv4.ipv4(4, header_length, tos, total_length, identification, 0, 0, VRRP_IPV4_TTL, inet.IPPROTO_VRRP, 0, primary_ip_address, VRRP_IPV4_DST_ADDRESS)
    p = packet.Packet()
    p.add_protocol(e)
    if vlan_id is not None:
        vlan_ = vlan.vlan(0, 0, vlan_id, e.ethertype)
        e.ethertype = ether.ETH_TYPE_8021Q
        p.add_protocol(vlan_)
    p.add_protocol(ip)
    p.add_protocol(self)
    return p