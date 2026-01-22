from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_check_enabled_false(self, command):
    """Check enabled returns False

        Verifies enabled returns False on empty alembic_ini_path variable
        """
    self.migration_config['alembic_ini_path'] = ''
    alembic = ext_alembic.AlembicExtension(self.engine, self.migration_config)
    self.assertFalse(alembic.enabled)