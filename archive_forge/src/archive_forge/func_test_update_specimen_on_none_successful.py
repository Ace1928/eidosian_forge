from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_specimen_on_none_successful(self):
    uuid = 'bdf3893c-ee3c-40a0-bc79-960adb6cd1d4'
    specimen = MyModel(y='y2', z=None, uuid=uuid)
    result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'})
    self.assertIn(result, self.session)
    self.assertEqual(uuid, result.uuid)
    self.assertEqual(5, result.id)
    self.assertEqual('z3', result.z)
    self._assert_row(5, {'uuid': 'bdf3893c-ee3c-40a0-bc79-960adb6cd1d4', 'x': 9, 'y': 'y2', 'z': 'z3'})