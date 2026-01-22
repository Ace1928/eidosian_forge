from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_object_namespace_not_found(self):
    object = _db_object_fixture(name='added_object')
    self.assertEqual('added_object', object['name'])
    self.assertRaises(exception.NotFound, self.db.metadef_object_create, self.context, 'not-a-namespace', object)