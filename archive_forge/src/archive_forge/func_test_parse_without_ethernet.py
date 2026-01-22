import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import lldp
from os_ken.lib import addrconv
def test_parse_without_ethernet(self):
    buf = self.data[ethernet.ethernet._MIN_LEN:]
    lldp_pkt, cls, rest_buf = lldp.lldp.parser(buf)
    self.assertEqual(len(rest_buf), 0)
    tlvs = lldp_pkt.tlvs
    self.assertEqual(tlvs[0].tlv_type, lldp.LLDP_TLV_CHASSIS_ID)
    self.assertEqual(tlvs[0].len, 7)
    self.assertEqual(tlvs[0].subtype, lldp.ChassisID.SUB_MAC_ADDRESS)
    self.assertEqual(tlvs[0].chassis_id, b'\x00\x04\x96\x1f\xa7&')
    self.assertEqual(tlvs[1].tlv_type, lldp.LLDP_TLV_PORT_ID)
    self.assertEqual(tlvs[1].len, 4)
    self.assertEqual(tlvs[1].subtype, lldp.PortID.SUB_INTERFACE_NAME)
    self.assertEqual(tlvs[1].port_id, b'1/3')
    self.assertEqual(tlvs[2].tlv_type, lldp.LLDP_TLV_TTL)
    self.assertEqual(tlvs[2].len, 2)
    self.assertEqual(tlvs[2].ttl, 120)
    self.assertEqual(tlvs[3].tlv_type, lldp.LLDP_TLV_END)