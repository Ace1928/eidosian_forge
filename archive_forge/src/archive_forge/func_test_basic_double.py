import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
def test_basic_double(self):
    user = ('duration', (100, 200))
    on_wire = b'\x80\x02\x00\x08\x00\x00\x00d\x00\x00\x00\xc8'
    self._test(user, on_wire, 4)