import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
def test_rollback_doesnt_interfere_with_killed_conn(self):
    session = self.sessionmaker()
    session.begin()
    try:
        session.execute(sql.text('select 1'))
        compat.driver_connection(session.connection()).close()
        session.execute(sql.text('select 1'))
    except exception.DBConnectionError:
        session.rollback()
    else:
        assert False, 'no exception raised'