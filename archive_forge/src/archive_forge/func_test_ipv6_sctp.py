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
def test_ipv6_sctp(self):
    e = ethernet.ethernet(ethertype=ether.ETH_TYPE_IPV6)
    ip = ipv6.ipv6(nxt=inet.IPPROTO_SCTP)
    s = sctp.sctp(chunks=[sctp.chunk_data(payload_data=self.payload)])
    p = e / ip / s
    p.serialize()
    ipaddr = addrconv.ipv6.text_to_bin('::')
    e_buf = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x86\xdd'
    ip_buf = b'`\x00\x00\x00' + b'\x00\x00' + b'\x84' + b'\xff' + b'\x00\x00' + ipaddr + ipaddr
    s_buf = b'\x00\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x00\x00\x00' + b'\x00' + b'\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + self.payload
    buf = e_buf + ip_buf + s_buf
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv6 = protocols['ipv6']
    p_sctp = protocols['sctp']
    self.assertTrue(p_eth)
    self.assertEqual('ff:ff:ff:ff:ff:ff', p_eth.dst)
    self.assertEqual('00:00:00:00:00:00', p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IPV6, p_eth.ethertype)
    self.assertTrue(p_ipv6)
    self.assertEqual(6, p_ipv6.version)
    self.assertEqual(0, p_ipv6.traffic_class)
    self.assertEqual(0, p_ipv6.flow_label)
    self.assertEqual(len(s_buf), p_ipv6.payload_length)
    self.assertEqual(inet.IPPROTO_SCTP, p_ipv6.nxt)
    self.assertEqual(255, p_ipv6.hop_limit)
    self.assertEqual('10::10', p_ipv6.src)
    self.assertEqual('20::20', p_ipv6.dst)
    self.assertTrue(p_sctp)
    self.assertEqual(1, p_sctp.src_port)
    self.assertEqual(1, p_sctp.dst_port)
    self.assertEqual(0, p_sctp.vtag)
    assert isinstance(p_sctp.chunks[0], sctp.chunk_data)
    self.assertEqual(0, p_sctp.chunks[0]._type)
    self.assertEqual(0, p_sctp.chunks[0].unordered)
    self.assertEqual(0, p_sctp.chunks[0].begin)
    self.assertEqual(0, p_sctp.chunks[0].end)
    self.assertEqual(16 + len(self.payload), p_sctp.chunks[0].length)
    self.assertEqual(0, p_sctp.chunks[0].tsn)
    self.assertEqual(0, p_sctp.chunks[0].sid)
    self.assertEqual(0, p_sctp.chunks[0].seq)
    self.assertEqual(0, p_sctp.chunks[0].payload_id)
    self.assertEqual(self.payload, p_sctp.chunks[0].payload_data)
    self.assertEqual(len(s_buf), len(p_sctp))
    eth_values = {'dst': 'ff:ff:ff:ff:ff:ff', 'src': '00:00:00:00:00:00', 'ethertype': ether.ETH_TYPE_IPV6}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv6_values = {'version': 6, 'traffic_class': 0, 'flow_label': 0, 'payload_length': len(s_buf), 'nxt': inet.IPPROTO_SCTP, 'hop_limit': 255, 'src': '10::10', 'dst': '20::20', 'ext_hdrs': []}
    _ipv6_str = ','.join(['%s=%s' % (k, repr(ipv6_values[k])) for k, v in inspect.getmembers(p_ipv6) if k in ipv6_values])
    ipv6_str = '%s(%s)' % (ipv6.ipv6.__name__, _ipv6_str)
    data_values = {'unordered': 0, 'begin': 0, 'end': 0, 'length': 16 + len(self.payload), 'tsn': 0, 'sid': 0, 'seq': 0, 'payload_id': 0, 'payload_data': self.payload}
    _data_str = ','.join(['%s=%s' % (k, repr(data_values[k])) for k in sorted(data_values.keys())])
    data_str = '[%s(%s)]' % (sctp.chunk_data.__name__, _data_str)
    sctp_values = {'src_port': 1, 'dst_port': 1, 'vtag': 0, 'csum': repr(p_sctp.csum), 'chunks': data_str}
    _sctp_str = ','.join(['%s=%s' % (k, sctp_values[k]) for k, _ in inspect.getmembers(p_sctp) if k in sctp_values])
    sctp_str = '%s(%s)' % (sctp.sctp.__name__, _sctp_str)
    pkt_str = '%s, %s, %s' % (eth_str, ipv6_str, sctp_str)
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv6_str, str(p_ipv6))
    self.assertEqual(ipv6_str, repr(p_ipv6))
    self.assertEqual(sctp_str, str(p_sctp))
    self.assertEqual(sctp_str, repr(p_sctp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))