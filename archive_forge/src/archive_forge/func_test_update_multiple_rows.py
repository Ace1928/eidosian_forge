from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_multiple_rows(self):
    exc = self.assertRaises(update_match.MultiRowsMatched, self.session.query(MyModel).filter_by(y='y1', z='z1').update_returning_pk, {'x': 9, 'z': 'z3'}, ('y', 'y1'))
    self.assertEqual('2 rows matched; expected one', exc.args[0])