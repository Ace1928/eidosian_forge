from alembic import command as alembic_api
from alembic import script as alembic_script
import fixtures
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from oslotest import base as test_base
import sqlalchemy
import testtools
from heat.db import migration
from heat.db import models
def test_single_head_revision(self):
    """Ensure we only have a single head revision.

        There's no good reason for us to have diverging history, so validate
        that only one head revision exists. This will prevent merge conflicts
        adding additional head revision points. If this fail for your change,
        look for migrations with the same 'revises' line in them.
        """
    script = alembic_script.ScriptDirectory.from_config(self.config)
    self.assertEqual(1, len(script.get_heads()))