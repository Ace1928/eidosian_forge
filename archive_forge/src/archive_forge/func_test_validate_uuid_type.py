import re
import unittest
from wsme import exc
from wsme import types
def test_validate_uuid_type(self):
    v = types.UuidType()
    self.assertEqual(v.validate('6a0a707c-45ef-4758-b533-e55adddba8ce'), '6a0a707c-45ef-4758-b533-e55adddba8ce')
    self.assertEqual(v.validate('6a0a707c45ef4758b533e55adddba8ce'), '6a0a707c-45ef-4758-b533-e55adddba8ce')
    self.assertRaises(ValueError, v.validate, '')
    self.assertRaises(ValueError, v.validate, 'foo')
    self.assertRaises(ValueError, v.validate, '6a0a707c-45ef-4758-b533-e55adddba8ce-a')