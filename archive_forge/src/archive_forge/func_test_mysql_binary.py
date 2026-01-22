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
def test_mysql_binary(self):
    self._run_dupe_constraint_test('mysql', '(1062, \'Duplicate entry \\\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E\\\' for key \\\'PRIMARY\\\'\')', expected_columns=['PRIMARY'], expected_value='\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E')
    self._run_dupe_constraint_test('mysql', '(1062, \'Duplicate entry \'\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E!,\' for key \'PRIMARY\'\')', expected_columns=['PRIMARY'], expected_value='\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E!,')