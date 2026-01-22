from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_merge_properties(self):
    self.schema.merge_properties({'bacon': {'type': 'string'}})
    expected = set(['ham', 'eggs', 'bacon'])
    actual = set(self.schema.raw()['properties'].keys())
    self.assertEqual(expected, actual)