from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
@mock.patch('glance.db.sqlalchemy.api.purge_deleted_rows')
def test_purge_command_fk_constraint_failure(self, purge_deleted_rows):
    purge_deleted_rows.side_effect = db_exception.DBReferenceError('fake_table', 'fake_constraint', 'fake_key', 'fake_key_table')
    exit = self.assertRaises(SystemExit, self.commands.purge, 10, 100)
    self.assertEqual('Purge command failed, check glance-manage logs for more details.', exit.code)