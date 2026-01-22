import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_schema_with_property(self):
    raw_schema = {'name': 'Country', 'properties': {'size': {}}}
    schema = schemas.Schema(raw_schema)
    self.assertEqual('Country', schema.name)
    self.assertEqual(['size'], [p.name for p in schema.properties])