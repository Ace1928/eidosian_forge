from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_tag(self):
    tags = self.tag_repo.list(filters={'namespace': NAMESPACE1})
    tag_names = set([t.name for t in tags])
    self.assertEqual(set([TAG1, TAG2, TAG3]), tag_names)