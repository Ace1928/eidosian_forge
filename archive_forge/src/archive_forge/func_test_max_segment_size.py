import re
import unittest
from oslo_config import types
def test_max_segment_size(self):
    self.assertConvertedEqual('host.%s.com' % ('x' * 63))
    self.assertInvalid('host.%s.com' % ('x' * 64))