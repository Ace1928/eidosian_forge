from unittest import mock
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db.sqlalchemy import test_migrations as migrate
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_models_not_sync_filtered(self):
    self._test_models_not_sync_filtered()