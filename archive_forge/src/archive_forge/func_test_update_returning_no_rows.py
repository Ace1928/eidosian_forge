from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_returning_no_rows(self):
    exc = self.assertRaises(update_match.NoRowsMatched, self.session.query(MyModel).filter_by(y='y1', z='z3').update_returning_pk, {'x': 9, 'z': 'z3'}, ('uuid', '136254d5-3869-408f-9da7-190e0072641a'))
    self.assertEqual('No rows matched the UPDATE', exc.args[0])