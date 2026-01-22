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
def test_ipv6_udp(self):
    e = ethernet.ethernet(ethertype=ether.ETH_TYPE_IPV6)
    ip = ipv6.ipv6(nxt=inet.IPPROTO_UDP)
    u = udp.udp()
    p = e / ip / u / self.payload
    p.serialize()
    ipaddr = addrconv.ipv6.text_to_bin('::')
    e_buf = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x86\xdd'
    ip_buf = b'`\x00\x00\x00' + b'\x00\x00' + b'\x11' + b'\xff' + b'\x00\x00' + ipaddr + ipaddr
    u_buf = b'\x00\x00' + b'\x00\x00' + b'\x00(' + b'\x00\x00'
    buf = e_buf + ip_buf + u_buf + self.payload
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv6 = protocols['ipv6']
    p_udp = protocols['udp']
    self.assertTrue(p_eth)
    self.assertEqual('ff:ff:ff:ff:ff:ff', p_eth.dst)
    self.assertEqual('00:00:00:00:00:00', p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IPV6, p_eth.ethertype)
    self.assertTrue(p_ipv6)
    self.assertEqual(6, p_ipv6.version)
    self.assertEqual(0, p_ipv6.traffic_class)
    self.assertEqual(0, p_ipv6.flow_label)
    self.assertEqual(len(u_buf) + len(self.payload), p_ipv6.payload_length)
    self.assertEqual(inet.IPPROTO_UDP, p_ipv6.nxt)
    self.assertEqual(255, p_ipv6.hop_limit)
    self.assertEqual('10::10', p_ipv6.src)
    self.assertEqual('20::20', p_ipv6.dst)
    self.assertTrue(p_udp)
    self.assertEqual(1, p_udp.src_port)
    self.assertEqual(1, p_udp.dst_port)
    self.assertEqual(len(u_buf) + len(self.payload), p_udp.total_length)
    self.assertEqual(11104, p_udp.csum)
    t = bytearray(u_buf)
    struct.pack_into('!H', t, 6, p_udp.csum)
    ph = struct.pack('!16s16sI3xB', ipaddr, ipaddr, len(u_buf) + len(self.payload), 17)
    t = ph + t + self.payload
    self.assertEqual(packet_utils.checksum(t), 98)
    self.assertTrue('payload' in protocols)
    self.assertEqual(self.payload, protocols['payload'])
    eth_values = {'dst': 'ff:ff:ff:ff:ff:ff', 'src': '00:00:00:00:00:00', 'ethertype': ether.ETH_TYPE_IPV6}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv6_values = {'version': 6, 'traffic_class': 0, 'flow_label': 0, 'payload_length': len(u_buf) + len(self.payload), 'nxt': inet.IPPROTO_UDP, 'hop_limit': 255, 'src': '10::10', 'dst': '20::20', 'ext_hdrs': []}
    _ipv6_str = ','.join(['%s=%s' % (k, repr(ipv6_values[k])) for k, v in inspect.getmembers(p_ipv6) if k in ipv6_values])
    ipv6_str = '%s(%s)' % (ipv6.ipv6.__name__, _ipv6_str)
    udp_values = {'src_port': 1, 'dst_port': 1, 'total_length': len(u_buf) + len(self.payload), 'csum': 11104}
    _udp_str = ','.join(['%s=%s' % (k, repr(udp_values[k])) for k, v in inspect.getmembers(p_udp) if k in udp_values])
    udp_str = '%s(%s)' % (udp.udp.__name__, _udp_str)
    pkt_str = '%s, %s, %s, %s' % (eth_str, ipv6_str, udp_str, repr(protocols['payload']))
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv6_str, str(p_ipv6))
    self.assertEqual(ipv6_str, repr(p_ipv6))
    self.assertEqual(udp_str, str(p_udp))
    self.assertEqual(udp_str, repr(p_udp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))