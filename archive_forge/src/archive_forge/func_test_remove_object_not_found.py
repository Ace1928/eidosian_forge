from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_remove_object_not_found(self):
    fake_name = 'fake_name'
    object = self.object_repo.get(NAMESPACE1, OBJECT1)
    object.name = fake_name
    self.assertRaises(exception.NotFound, self.object_repo.remove, object)