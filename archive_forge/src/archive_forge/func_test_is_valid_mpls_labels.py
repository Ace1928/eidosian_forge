import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_mpls_labels(self):
    self.assertTrue(validation.is_valid_mpls_labels([100, 200]))