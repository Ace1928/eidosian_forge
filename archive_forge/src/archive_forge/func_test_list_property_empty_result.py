from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_property_empty_result(self):
    properties = self.property_repo.list(filters={'namespace': NAMESPACE2})
    property_names = set([p.name for p in properties])
    self.assertEqual(set([]), property_names)