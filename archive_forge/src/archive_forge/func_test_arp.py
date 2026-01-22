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
def test_arp(self):
    e = ethernet.ethernet(self.dst_mac, self.src_mac, ether.ETH_TYPE_ARP)
    a = arp.arp(1, ether.ETH_TYPE_IP, 6, 4, 2, self.src_mac, self.src_ip, self.dst_mac, self.dst_ip)
    p = packet.Packet()
    p.add_protocol(e)
    p.add_protocol(a)
    p.serialize()
    e_buf = self.dst_mac_bin + self.src_mac_bin + b'\x08\x06'
    a_buf = b'\x00\x01' + b'\x08\x00' + b'\x06' + b'\x04' + b'\x00\x02' + self.src_mac_bin + self.src_ip_bin + self.dst_mac_bin + self.dst_ip_bin
    buf = e_buf + a_buf
    pad_len = 60 - len(buf)
    if pad_len > 0:
        buf += b'\x00' * pad_len
    self.assertEqual(buf, p.data)
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_arp = protocols['arp']
    self.assertTrue(p_eth)
    self.assertEqual(self.dst_mac, p_eth.dst)
    self.assertEqual(self.src_mac, p_eth.src)
    self.assertEqual(ether.ETH_TYPE_ARP, p_eth.ethertype)
    self.assertTrue(p_arp)
    self.assertEqual(1, p_arp.hwtype)
    self.assertEqual(ether.ETH_TYPE_IP, p_arp.proto)
    self.assertEqual(6, p_arp.hlen)
    self.assertEqual(4, p_arp.plen)
    self.assertEqual(2, p_arp.opcode)
    self.assertEqual(self.src_mac, p_arp.src_mac)
    self.assertEqual(self.src_ip, p_arp.src_ip)
    self.assertEqual(self.dst_mac, p_arp.dst_mac)
    self.assertEqual(self.dst_ip, p_arp.dst_ip)
    eth_values = {'dst': self.dst_mac, 'src': self.src_mac, 'ethertype': ether.ETH_TYPE_ARP}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    arp_values = {'hwtype': 1, 'proto': ether.ETH_TYPE_IP, 'hlen': 6, 'plen': 4, 'opcode': 2, 'src_mac': self.src_mac, 'dst_mac': self.dst_mac, 'src_ip': self.src_ip, 'dst_ip': self.dst_ip}
    _arp_str = ','.join(['%s=%s' % (k, repr(arp_values[k])) for k, v in inspect.getmembers(p_arp) if k in arp_values])
    arp_str = '%s(%s)' % (arp.arp.__name__, _arp_str)
    pkt_str = '%s, %s' % (eth_str, arp_str)
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(arp_str, str(p_arp))
    self.assertEqual(arp_str, repr(p_arp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))