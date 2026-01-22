from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
def test_purge_invalid_max_rows(self):
    max_rows = 'abcd'
    ex = self.assertRaises(SystemExit, self.commands.purge, 1, max_rows)
    expected = 'Invalid int value for max_rows: %(max_rows)s' % {'max_rows': max_rows}
    self.assertEqual(expected, ex.code)