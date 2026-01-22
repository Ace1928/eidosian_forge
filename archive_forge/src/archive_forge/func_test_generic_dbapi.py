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
def test_generic_dbapi(self):
    matched = self._run_test('mysql', 'select you_made_a_programming_error', self.ProgrammingError('Error 123, you made a mistake'), exception.DBError)
    self.assertInnerException(matched, 'ProgrammingError', 'Error 123, you made a mistake', 'select you_made_a_programming_error', ())