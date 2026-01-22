import uuid
from oslotest import base as test_base
from oslo_utils import uuidutils
def test_generate_uuid(self):
    uuid_string = uuidutils.generate_uuid()
    self.assertIsInstance(uuid_string, str)
    self.assertEqual(len(uuid_string), 36)
    self.assertEqual(len(uuid_string.replace('-', '')), 32)