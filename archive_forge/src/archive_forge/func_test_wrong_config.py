from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_wrong_config(self):
    err = self.assertRaises(ValueError, manager.MigrationManager, {'wrong_key': 'sqlite://'})
    self.assertEqual('Either database url or engine must be provided.', err.args[0])