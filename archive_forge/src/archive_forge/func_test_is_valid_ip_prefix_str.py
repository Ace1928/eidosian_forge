import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ip_prefix_str(self):
    self.assertTrue(validation.is_valid_ip_prefix('24', 32))