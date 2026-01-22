from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_returning_pk_matched(self):
    pk = self.session.query(MyModel).filter_by(y='y1', z='z2').update_returning_pk({'x': 9, 'z': 'z3'}, ('uuid', '136254d5-3869-408f-9da7-190e0072641a'))
    self.assertEqual((2,), pk)
    self._assert_row(2, {'uuid': '136254d5-3869-408f-9da7-190e0072641a', 'x': 9, 'y': 'y1', 'z': 'z3'})