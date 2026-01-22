from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
@mock.patch('importlib.import_module')
@mock.patch('pkgutil.iter_modules')
def test_find_migrations(self, mock_iter, mock_import):

    def fake_iter_modules(blah):
        yield ('blah', 'zebra01', 'blah')
        yield ('blah', 'zebra02', 'blah')
        yield ('blah', 'yellow01', 'blah')
        yield ('blah', 'xray01', 'blah')
        yield ('blah', 'wrinkle01', 'blah')
    mock_iter.side_effect = fake_iter_modules
    zebra1 = mock.Mock()
    zebra1.has_migrations.return_value = mock.Mock()
    zebra1.migrate.return_value = mock.Mock()
    zebra2 = mock.Mock()
    zebra2.has_migrations.return_value = mock.Mock()
    zebra2.migrate.return_value = mock.Mock()
    fake_imported_modules = [zebra1, zebra2]
    mock_import.side_effect = fake_imported_modules
    actual = data_migrations._find_migration_modules('zebra')
    self.assertEqual(2, len(actual))
    self.assertEqual(fake_imported_modules, actual)