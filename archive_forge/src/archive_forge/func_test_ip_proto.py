import unittest
import logging
from os_ken.ofproto.inet import *
def test_ip_proto(self):
    self.assertEqual(IPPROTO_IP, 0)
    self.assertEqual(IPPROTO_HOPOPTS, 0)
    self.assertEqual(IPPROTO_ICMP, 1)
    self.assertEqual(IPPROTO_TCP, 6)
    self.assertEqual(IPPROTO_UDP, 17)
    self.assertEqual(IPPROTO_ROUTING, 43)
    self.assertEqual(IPPROTO_FRAGMENT, 44)
    self.assertEqual(IPPROTO_AH, 51)
    self.assertEqual(IPPROTO_ICMPV6, 58)
    self.assertEqual(IPPROTO_NONE, 59)
    self.assertEqual(IPPROTO_DSTOPTS, 60)
    self.assertEqual(IPPROTO_SCTP, 132)