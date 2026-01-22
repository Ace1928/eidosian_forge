import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
def test_build_mldv2_query(self):
    p = self._build_mldv2_query()
    e = self.find_protocol(p, 'ethernet')
    self.assertTrue(e)
    self.assertEqual(e.ethertype, ether.ETH_TYPE_IPV6)
    i = self.find_protocol(p, 'ipv6')
    self.assertTrue(i)
    self.assertEqual(i.nxt, inet.IPPROTO_ICMPV6)
    ic = self.find_protocol(p, 'icmpv6')
    self.assertTrue(ic)
    self.assertEqual(ic.type_, icmpv6.MLD_LISTENER_QUERY)
    self.assertEqual(ic.data.maxresp, self.maxresp)
    self.assertEqual(ic.data.address, self.address)
    self.assertEqual(ic.data.s_flg, self.s_flg)
    self.assertEqual(ic.data.qrv, self.qrv)
    self.assertEqual(ic.data.num, self.num)
    self.assertEqual(ic.data.srcs, self.srcs)