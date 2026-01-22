import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_schema_minimum(self):
    raw_schema = {'name': 'Country', 'properties': {}}
    schema = schemas.Schema(raw_schema)
    self.assertEqual('Country', schema.name)
    self.assertEqual([], schema.properties)