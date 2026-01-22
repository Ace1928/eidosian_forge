import logging
import struct
import unittest
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import vlan
from os_ken.lib.packet import pbb
def test_build_itag(self):
    p = self._build_itag()
    e = p.get_protocols(ethernet.ethernet)
    self.assertTrue(e)
    self.assertTrue(isinstance(e, list))
    self.assertEqual(e[0].ethertype, ether.ETH_TYPE_8021AD)
    self.assertEqual(e[1].ethertype, ether.ETH_TYPE_8021AD)
    sv = p.get_protocols(vlan.svlan)
    self.assertTrue(sv)
    self.assertTrue(isinstance(sv, list))
    self.assertEqual(sv[0].ethertype, ether.ETH_TYPE_8021Q)
    self.assertEqual(sv[1].ethertype, ether.ETH_TYPE_8021Q)
    it = p.get_protocol(pbb.itag)
    self.assertTrue(it)
    v = p.get_protocol(vlan.vlan)
    self.assertTrue(v)
    self.assertEqual(v.ethertype, ether.ETH_TYPE_IP)
    ip = p.get_protocol(ipv4.ipv4)
    self.assertTrue(ip)
    self.assertEqual(it.pcp, self.pcp)
    self.assertEqual(it.dei, self.dei)
    self.assertEqual(it.uca, self.uca)
    self.assertEqual(it.sid, self.sid)