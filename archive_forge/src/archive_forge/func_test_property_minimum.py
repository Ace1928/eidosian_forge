import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_property_minimum(self):
    prop = schemas.SchemaProperty('size')
    self.assertEqual('size', prop.name)