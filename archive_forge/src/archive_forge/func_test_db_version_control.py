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
@mock.patch.object(manage.DbCommands, 'version_control')
def test_db_version_control(self, version_control):
    self._main_test_helper(['glance.cmd.manage', 'db', 'version_control'], manage.DbCommands.version_control)