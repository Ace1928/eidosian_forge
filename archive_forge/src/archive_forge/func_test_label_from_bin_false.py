import unittest
import logging
import inspect
from os_ken.lib.packet import mpls
def test_label_from_bin_false(self):
    mpls_label = 1048575
    is_bos = False
    buf = b'\xff\xff\xf0'
    mpls_label_out, is_bos_out = mpls.label_from_bin(buf)
    self.assertEqual(mpls_label, mpls_label_out)
    self.assertEqual(is_bos, is_bos_out)