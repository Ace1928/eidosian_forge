from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_property_namespace_not_found(self):
    property = _db_property_fixture(name='added_property')
    self.assertEqual('added_property', property['name'])
    self.assertRaises(exception.NotFound, self.db.metadef_property_create, self.context, 'not_a_namespace', property)