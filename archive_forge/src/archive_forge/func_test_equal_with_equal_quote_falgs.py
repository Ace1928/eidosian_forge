import re
import unittest
from oslo_config import types
def test_equal_with_equal_quote_falgs(self):
    t1 = types.String(quotes=True)
    t2 = types.String(quotes=True)
    self.assertTrue(t1 == t2)