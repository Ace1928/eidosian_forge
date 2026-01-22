import unittest
import logging
import inspect
from os_ken.lib.packet import mpls
def test_label_to_bin_true(self):
    mpls_label = 1048575
    is_bos = True
    label = b'\xff\xff\xf1'
    label_out = mpls.label_to_bin(mpls_label, is_bos)
    self.assertEqual(label, label_out)