import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_old_asn_over(self):
    self.assertEqual(False, validation.is_valid_old_asn(65535 + 1))