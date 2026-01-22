from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_filter_passes_extra_properties(self):
    obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
    filtered = self.schema.filter(obj)
    self.assertEqual(obj, filtered)