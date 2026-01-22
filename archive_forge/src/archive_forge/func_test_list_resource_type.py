from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_resource_type(self):
    resource_type = self.resource_type_repo.list(filters={'namespace': NAMESPACE1})
    self.assertEqual(0, len(resource_type))