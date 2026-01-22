from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_property(self):
    properties = self.property_repo.list(filters={'namespace': NAMESPACE1})
    property_names = set([p.name for p in properties])
    self.assertEqual(set([PROPERTY1, PROPERTY2, PROPERTY3]), property_names)