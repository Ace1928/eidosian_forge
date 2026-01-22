import unittest
from os_ken.lib.packet import ospf
def test_as_external_lsa(self):
    extnw1 = ospf.ASExternalLSA.ExternalNetwork(mask='255.255.255.0', metric=20, fwd_addr='10.0.0.1')
    msg = ospf.ASExternalLSA(id_='192.168.0.1', adv_router='192.168.0.2', extnws=[extnw1])
    binmsg = msg.serialize()
    msg2, cls, rest = ospf.LSA.parser(binmsg)
    self.assertEqual(msg.header.checksum, msg2.header.checksum)
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(rest, b'')