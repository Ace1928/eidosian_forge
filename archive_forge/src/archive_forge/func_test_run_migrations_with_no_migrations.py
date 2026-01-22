from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
def test_run_migrations_with_no_migrations(self):
    migrations = []
    actual = data_migrations._run_migrations(mock.Mock(), migrations)
    self.assertEqual(0, actual)