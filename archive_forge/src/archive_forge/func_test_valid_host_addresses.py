import re
import unittest
from oslo_config import types
def test_valid_host_addresses(self):
    self.assertConvertedValue('_foo', '_foo')
    self.assertConvertedValue('host_name', 'host_name')
    self.assertConvertedValue('overcloud-novacompute_edge1-0.internalapi.localdomain', 'overcloud-novacompute_edge1-0.internalapi.localdomain')
    self.assertConvertedValue('host_01.co.uk', 'host_01.co.uk')
    self.assertConvertedValue('_site01001', '_site01001')