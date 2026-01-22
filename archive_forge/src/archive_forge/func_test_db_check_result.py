import io
from unittest import mock
import fixtures
from glance.cmd import manage
from glance.common import exception
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata as db_metadata
from glance.tests import utils as test_utils
from sqlalchemy.engine.url import make_url as sqlalchemy_make_url
@mock.patch('glance.db.sqlalchemy.api.get_engine')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.data_migrations.has_pending_migrations')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_current_alembic_heads')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_alembic_branch_head')
def test_db_check_result(self, mock_get_alembic_branch_head, mock_get_current_alembic_heads, mock_has_pending_migrations, get_mock_engine):
    get_mock_engine.return_value = mock.Mock()
    engine = get_mock_engine.return_value
    engine.engine.name = 'postgresql'
    exit = self.assertRaises(SystemExit, self.db.check)
    self.assertIn('Rolling upgrades are currently supported only for MySQL and Sqlite', exit.code)
    engine = get_mock_engine.return_value
    engine.engine.name = 'mysql'
    mock_get_current_alembic_heads.return_value = ['ocata_contract01']
    mock_get_alembic_branch_head.return_value = 'pike_expand01'
    exit = self.assertRaises(SystemExit, self.db.check)
    self.assertEqual(3, exit.code)
    self.assertIn('Your database is not up to date. Your first step is to run `glance-manage db expand`.', self.output.getvalue())
    mock_get_current_alembic_heads.return_value = ['pike_expand01']
    mock_get_alembic_branch_head.side_effect = ['pike_expand01', None]
    mock_has_pending_migrations.return_value = [mock.Mock()]
    exit = self.assertRaises(SystemExit, self.db.check)
    self.assertEqual(4, exit.code)
    self.assertIn('Your database is not up to date. Your next step is to run `glance-manage db migrate`.', self.output.getvalue())
    mock_get_current_alembic_heads.return_value = ['pike_expand01']
    mock_get_alembic_branch_head.side_effect = ['pike_expand01', 'pike_contract01']
    mock_has_pending_migrations.return_value = None
    exit = self.assertRaises(SystemExit, self.db.check)
    self.assertEqual(5, exit.code)
    self.assertIn('Your database is not up to date. Your next step is to run `glance-manage db contract`.', self.output.getvalue())
    mock_get_current_alembic_heads.return_value = ['pike_contract01']
    mock_get_alembic_branch_head.side_effect = ['pike_expand01', 'pike_contract01']
    mock_has_pending_migrations.return_value = None
    self.assertRaises(SystemExit, self.db.check)
    self.assertIn('Database is up to date. No upgrades needed.', self.output.getvalue())