from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_save_namespace(self):
    namespace = self.namespace_repo.get(NAMESPACE1)
    namespace.display_name = 'save_name'
    namespace.description = 'save_desc'
    self.namespace_repo.save(namespace)
    namespace = self.namespace_repo.get(NAMESPACE1)
    self.assertEqual('save_name', namespace.display_name)
    self.assertEqual('save_desc', namespace.description)