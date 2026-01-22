import re
import unittest
from wsme import exc
from wsme import types
def test_validate_ipv6_address_type(self):
    v = types.IPv6AddressType()
    self.assertEqual(v.validate('0:0:0:0:0:0:0:1'), '0:0:0:0:0:0:0:1')
    self.assertEqual(v.validate(u'0:0:0:0:0:0:0:1'), u'0:0:0:0:0:0:0:1')
    self.assertEqual(v.validate('2001:0db8:bd05:01d2:288a:1fc0:0001:10ee'), '2001:0db8:bd05:01d2:288a:1fc0:0001:10ee')
    self.assertRaises(ValueError, v.validate, '')
    self.assertRaises(ValueError, v.validate, 'foo')
    self.assertRaises(ValueError, v.validate, '192.168.0.1')
    self.assertRaises(ValueError, v.validate, '0:0:0:0:0:0:1')