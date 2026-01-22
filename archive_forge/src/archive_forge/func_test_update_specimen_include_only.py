from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_specimen_include_only(self):
    uuid = '136254d5-3869-408f-9da7-190e0072641a'
    specimen = MyModel(y='y9', z='z5', x=6, uuid=uuid)
    self.session.query(MyModel).filter(MyModel.uuid == uuid).one()
    result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'}, include_only=('x',))
    self.assertEqual(uuid, result.uuid)
    self.assertEqual(2, result.id)
    self.assertEqual('z3', result.z)
    self.assertIn(result, self.session)
    self.assertNotIn(result, self.session.dirty)
    self._assert_row(2, {'uuid': '136254d5-3869-408f-9da7-190e0072641a', 'x': 9, 'y': 'y1', 'z': 'z3'})