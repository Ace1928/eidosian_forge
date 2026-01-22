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
def test_ipv4_sctp(self):
    e = ethernet.ethernet()
    ip = ipv4.ipv4(proto=inet.IPPROTO_SCTP)
    s = sctp.sctp(chunks=[sctp.chunk_data(payload_data=self.payload)])
    p = e / ip / s
    p.serialize()
    ipaddr = addrconv.ipv4.text_to_bin('0.0.0.0')
    e_buf = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x08\x00'
    ip_buf = b'E' + b'\x00' + b'\x00P' + b'\x00\x00' + b'\x00\x00' + b'\xff' + b'\x84' + b'\x00\x00' + ipaddr + ipaddr
    s_buf = b'\x00\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x00\x00\x00' + b'\x00' + b'\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + self.payload
    buf = e_buf + ip_buf + s_buf
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv4 = protocols['ipv4']
    p_sctp = protocols['sctp']
    self.assertTrue(p_eth)
    self.assertEqual('ff:ff:ff:ff:ff:ff', p_eth.dst)
    self.assertEqual('00:00:00:00:00:00', p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IP, p_eth.ethertype)
    self.assertTrue(p_ipv4)
    self.assertEqual(4, p_ipv4.version)
    self.assertEqual(5, p_ipv4.header_length)
    self.assertEqual(0, p_ipv4.tos)
    l = len(ip_buf) + len(s_buf)
    self.assertEqual(l, p_ipv4.total_length)
    self.assertEqual(0, p_ipv4.identification)
    self.assertEqual(0, p_ipv4.flags)
    self.assertEqual(255, p_ipv4.ttl)
    self.assertEqual(inet.IPPROTO_SCTP, p_ipv4.proto)
    self.assertEqual('10.0.0.1', p_ipv4.src)
    self.assertEqual('10.0.0.2', p_ipv4.dst)
    t = bytearray(ip_buf)
    struct.pack_into('!H', t, 10, p_ipv4.csum)
    self.assertEqual(packet_utils.checksum(t), 5123)
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
    eth_values = {'dst': 'ff:ff:ff:ff:ff:ff', 'src': '00:00:00:00:00:00', 'ethertype': ether.ETH_TYPE_IP}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv4_values = {'version': 4, 'header_length': 5, 'tos': 0, 'total_length': l, 'identification': 0, 'flags': 0, 'offset': 0, 'ttl': 255, 'proto': inet.IPPROTO_SCTP, 'csum': p_ipv4.csum, 'src': '10.0.0.1', 'dst': '10.0.0.2', 'option': None}
    _ipv4_str = ','.join(['%s=%s' % (k, repr(ipv4_values[k])) for k, v in inspect.getmembers(p_ipv4) if k in ipv4_values])
    ipv4_str = '%s(%s)' % (ipv4.ipv4.__name__, _ipv4_str)
    data_values = {'unordered': 0, 'begin': 0, 'end': 0, 'length': 16 + len(self.payload), 'tsn': 0, 'sid': 0, 'seq': 0, 'payload_id': 0, 'payload_data': self.payload}
    _data_str = ','.join(['%s=%s' % (k, repr(data_values[k])) for k in sorted(data_values.keys())])
    data_str = '[%s(%s)]' % (sctp.chunk_data.__name__, _data_str)
    sctp_values = {'src_port': 1, 'dst_port': 1, 'vtag': 0, 'csum': repr(p_sctp.csum), 'chunks': data_str}
    _sctp_str = ','.join(['%s=%s' % (k, sctp_values[k]) for k, _ in inspect.getmembers(p_sctp) if k in sctp_values])
    sctp_str = '%s(%s)' % (sctp.sctp.__name__, _sctp_str)
    pkt_str = '%s, %s, %s' % (eth_str, ipv4_str, sctp_str)
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv4_str, str(p_ipv4))
    self.assertEqual(ipv4_str, repr(p_ipv4))
    self.assertEqual(sctp_str, str(p_sctp))
    self.assertEqual(sctp_str, repr(p_sctp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))