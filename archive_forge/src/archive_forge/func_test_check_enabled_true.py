from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_check_enabled_true(self, command):
    """Check enabled returns True

        Verifies that enabled returns True on non empty
        alembic_ini_path conf variable
        """
    self.assertTrue(self.alembic.enabled)