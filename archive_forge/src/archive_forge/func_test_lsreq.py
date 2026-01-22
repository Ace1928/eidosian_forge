import unittest
from os_ken.lib.packet import ospf
def test_lsreq(self):
    req = ospf.OSPFLSReq.Request(type_=ospf.OSPF_ROUTER_LSA, id_='192.168.0.1', adv_router='192.168.0.2')
    msg = ospf.OSPFLSReq(router_id='192.168.0.1', lsa_requests=[req])
    binmsg = msg.serialize()
    msg2, cls, rest = ospf.OSPFMessage.parser(binmsg)
    self.assertEqual(msg.checksum, msg2.checksum)
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(rest, b'')