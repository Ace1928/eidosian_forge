import unittest
import logging
import struct
import inspect
from os_ken.ofproto import inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
def test_create_packet(self):
    primary_ip = '2001:db8:2000::3'
    p0 = self.vrrpv3.create_packet(primary_ip)
    p0.serialize()
    print(len(p0.data), p0.data)
    p1 = packet.Packet(bytes(p0.data))
    p1.serialize()
    print(len(p0.data), p0.data)
    print(len(p1.data), p1.data)
    self.assertEqual(p0.data, p1.data)