from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_private_namespaces(self):
    filters = {'visibility': 'private'}
    namespaces = self.namespace_repo.list(filters=filters)
    namespace_names = set([n.namespace for n in namespaces])
    self.assertEqual(set([NAMESPACE1]), namespace_names)