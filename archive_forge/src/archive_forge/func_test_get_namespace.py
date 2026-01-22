from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_namespace(self):
    namespace = self.namespace_repo.get(NAMESPACE1)
    self.assertEqual(NAMESPACE1, namespace.namespace)
    self.assertEqual('desc1', namespace.description)
    self.assertEqual('1', namespace.display_name)
    self.assertEqual(TENANT1, namespace.owner)
    self.assertTrue(namespace.protected)
    self.assertEqual('private', namespace.visibility)