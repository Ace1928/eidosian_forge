from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_object_empty_result(self):
    objects = self.object_repo.list(filters={'namespace': NAMESPACE2})
    object_names = set([o.name for o in objects])
    self.assertEqual(set([]), object_names)