import glance.api.v2.schemas
import glance.db.sqlalchemy.api as db_api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_member(self):
    req = unit_test_utils.get_fake_request()
    output = self.controller.member(req)
    self.assertEqual('member', output['name'])
    expected = set(['status', 'created_at', 'updated_at', 'image_id', 'member_id', 'schema'])
    self.assertEqual(expected, set(output['properties'].keys()))