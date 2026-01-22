from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_validate_with_additional_properties_allowed(self):
    obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
    self.schema.validate(obj)