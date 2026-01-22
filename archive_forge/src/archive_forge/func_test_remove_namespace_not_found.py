from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_remove_namespace_not_found(self):
    fake_name = 'fake_name'
    namespace = self.namespace_repo.get(NAMESPACE1)
    namespace.namespace = fake_name
    exc = self.assertRaises(exception.NotFound, self.namespace_repo.remove, namespace)
    self.assertIn(fake_name, encodeutils.exception_to_unicode(exc))