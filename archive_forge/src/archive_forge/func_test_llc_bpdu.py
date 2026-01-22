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
def test_llc_bpdu(self):
    e = ethernet.ethernet(self.dst_mac, self.src_mac, ether.ETH_TYPE_IEEE802_3)
    llc_control = llc.ControlFormatU(0, 0, 0)
    l = llc.llc(llc.SAP_BPDU, llc.SAP_BPDU, llc_control)
    b = bpdu.ConfigurationBPDUs(flags=0, root_priority=32768, root_system_id_extension=0, root_mac_address=self.src_mac, root_path_cost=0, bridge_priority=32768, bridge_system_id_extension=0, bridge_mac_address=self.dst_mac, port_priority=128, port_number=4, message_age=1, max_age=20, hello_time=2, forward_delay=15)
    p = packet.Packet()
    p.add_protocol(e)
    p.add_protocol(l)
    p.add_protocol(b)
    p.serialize()
    e_buf = self.dst_mac_bin + self.src_mac_bin + b'\x05\xdc'
    l_buf = b'BB\x03'
    b_buf = b'\x00\x00\x00\x00\x00\x80\x00\xbb\xbb\xbb\xbb\xbb\xbb\x00\x00\x00\x00\x80\x00\xaa\xaa\xaa\xaa\xaa\xaa\x80\x04\x01\x00\x14\x00\x02\x00\x0f\x00'
    buf = e_buf + l_buf + b_buf
    pad_len = 60 - len(buf)
    if pad_len > 0:
        buf += b'\x00' * pad_len
    self.assertEqual(buf, p.data)
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_llc = protocols['llc']
    p_bpdu = protocols['ConfigurationBPDUs']
    self.assertTrue(p_eth)
    self.assertEqual(self.dst_mac, p_eth.dst)
    self.assertEqual(self.src_mac, p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IEEE802_3, p_eth.ethertype)
    self.assertTrue(p_llc)
    self.assertEqual(llc.SAP_BPDU, p_llc.dsap_addr)
    self.assertEqual(llc.SAP_BPDU, p_llc.ssap_addr)
    self.assertEqual(0, p_llc.control.modifier_function1)
    self.assertEqual(0, p_llc.control.pf_bit)
    self.assertEqual(0, p_llc.control.modifier_function2)
    self.assertTrue(p_bpdu)
    self.assertEqual(bpdu.PROTOCOL_IDENTIFIER, p_bpdu._protocol_id)
    self.assertEqual(bpdu.PROTOCOLVERSION_ID_BPDU, p_bpdu._version_id)
    self.assertEqual(bpdu.TYPE_CONFIG_BPDU, p_bpdu._bpdu_type)
    self.assertEqual(0, p_bpdu.flags)
    self.assertEqual(32768, p_bpdu.root_priority)
    self.assertEqual(0, p_bpdu.root_system_id_extension)
    self.assertEqual(self.src_mac, p_bpdu.root_mac_address)
    self.assertEqual(0, p_bpdu.root_path_cost)
    self.assertEqual(32768, p_bpdu.bridge_priority)
    self.assertEqual(0, p_bpdu.bridge_system_id_extension)
    self.assertEqual(self.dst_mac, p_bpdu.bridge_mac_address)
    self.assertEqual(128, p_bpdu.port_priority)
    self.assertEqual(4, p_bpdu.port_number)
    self.assertEqual(1, p_bpdu.message_age)
    self.assertEqual(20, p_bpdu.max_age)
    self.assertEqual(2, p_bpdu.hello_time)
    self.assertEqual(15, p_bpdu.forward_delay)
    eth_values = {'dst': self.dst_mac, 'src': self.src_mac, 'ethertype': ether.ETH_TYPE_IEEE802_3}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ctrl_values = {'modifier_function1': 0, 'pf_bit': 0, 'modifier_function2': 0}
    _ctrl_str = ','.join(['%s=%s' % (k, repr(ctrl_values[k])) for k, v in inspect.getmembers(p_llc.control) if k in ctrl_values])
    ctrl_str = '%s(%s)' % (llc.ControlFormatU.__name__, _ctrl_str)
    llc_values = {'dsap_addr': repr(llc.SAP_BPDU), 'ssap_addr': repr(llc.SAP_BPDU), 'control': ctrl_str}
    _llc_str = ','.join(['%s=%s' % (k, llc_values[k]) for k, v in inspect.getmembers(p_llc) if k in llc_values])
    llc_str = '%s(%s)' % (llc.llc.__name__, _llc_str)
    bpdu_values = {'flags': 0, 'root_priority': int(32768), 'root_system_id_extension': int(0), 'root_mac_address': self.src_mac, 'root_path_cost': 0, 'bridge_priority': int(32768), 'bridge_system_id_extension': int(0), 'bridge_mac_address': self.dst_mac, 'port_priority': 128, 'port_number': 4, 'message_age': float(1), 'max_age': float(20), 'hello_time': float(2), 'forward_delay': float(15)}
    _bpdu_str = ','.join(['%s=%s' % (k, repr(bpdu_values[k])) for k, v in inspect.getmembers(p_bpdu) if k in bpdu_values])
    bpdu_str = '%s(%s)' % (bpdu.ConfigurationBPDUs.__name__, _bpdu_str)
    pkt_str = '%s, %s, %s' % (eth_str, llc_str, bpdu_str)
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(llc_str, str(p_llc))
    self.assertEqual(llc_str, repr(p_llc))
    self.assertEqual(bpdu_str, str(p_bpdu))
    self.assertEqual(bpdu_str, repr(p_bpdu))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))