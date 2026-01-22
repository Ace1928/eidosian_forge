from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
@mock.patch('glance.db.sqlalchemy.alembic_migrations.data_migrations._find_migration_modules')
def test_has_pending_migrations_no_migrations(self, mock_find):
    mock_find.return_value = None
    self.assertFalse(data_migrations.has_pending_migrations(mock.Mock()))