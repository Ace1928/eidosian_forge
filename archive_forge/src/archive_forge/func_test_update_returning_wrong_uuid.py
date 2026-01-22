from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_returning_wrong_uuid(self):
    exc = self.assertRaises(update_match.NoRowsMatched, self.session.query(MyModel).filter_by(y='y1', z='z2').update_returning_pk, {'x': 9, 'z': 'z3'}, ('uuid', '23cb9224-9f8e-40fe-bd3c-e7577b7af37d'))
    self.assertEqual('No rows matched the UPDATE', exc.args[0])