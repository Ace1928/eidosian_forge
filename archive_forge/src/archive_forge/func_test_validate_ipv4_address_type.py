import re
import unittest
from wsme import exc
from wsme import types
def test_validate_ipv4_address_type(self):
    v = types.IPv4AddressType()
    self.assertEqual(v.validate('127.0.0.1'), '127.0.0.1')
    self.assertEqual(v.validate('192.168.0.1'), '192.168.0.1')
    self.assertEqual(v.validate(u'8.8.1.1'), u'8.8.1.1')
    self.assertRaises(ValueError, v.validate, '')
    self.assertRaises(ValueError, v.validate, 'foo')
    self.assertRaises(ValueError, v.validate, '2001:0db8:bd05:01d2:288a:1fc0:0001:10ee')
    self.assertRaises(ValueError, v.validate, '1.2.3')