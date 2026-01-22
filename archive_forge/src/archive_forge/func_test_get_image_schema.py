from openstack.image.v2 import schema as _schema
from openstack.tests.functional.image.v2 import base
def test_get_image_schema(self):
    schema = self.conn.image.get_image_schema()
    self.assertIsNotNone(schema)
    self.assertIsInstance(schema, _schema.Schema)