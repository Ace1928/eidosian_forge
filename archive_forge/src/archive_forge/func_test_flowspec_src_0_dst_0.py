import unittest
import os_ken.ofproto.ofproto_v1_3_parser as ofpp
def test_flowspec_src_0_dst_0(self):
    user = ofpp.NXFlowSpecMatch(src=('in_port', 0), dst=('in_port', 0), n_bits=16)
    on_wire = b'\x00\x10\x80\x00\x00\x04\x00\x00\x80\x00\x00\x04\x00\x00'
    self.assertEqual(on_wire, user.serialize())
    o, rest = ofpp._NXFlowSpec.parse(on_wire)
    self.assertEqual(user.to_jsondict(), o.to_jsondict())
    self.assertEqual(str(user), str(o))
    self.assertEqual(b'', rest)