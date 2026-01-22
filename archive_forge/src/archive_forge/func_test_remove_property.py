from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_remove_property(self):
    property = self.property_repo.get(NAMESPACE1, PROPERTY1)
    self.property_repo.remove(property)
    self.assertRaises(exception.NotFound, self.property_repo.get, NAMESPACE1, PROPERTY1)