import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet import arp
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import icmp, icmpv6
from os_ken.lib.packet import ipv4, ipv6
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet, packet_utils
from os_ken.lib.packet import sctp
from os_ken.lib.packet import tcp, udp
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
def test_div_api(self):
    e = ethernet.ethernet(self.dst_mac, self.src_mac, ether.ETH_TYPE_IP)
    i = ipv4.ipv4()
    u = udp.udp(self.src_port, self.dst_port)
    pkt = e / i / u
    self.assertTrue(isinstance(pkt, packet.Packet))
    self.assertTrue(isinstance(pkt.protocols[0], ethernet.ethernet))
    self.assertTrue(isinstance(pkt.protocols[1], ipv4.ipv4))
    self.assertTrue(isinstance(pkt.protocols[2], udp.udp))