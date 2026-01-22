import os
from unittest import mock
from sqlalchemy.engine import url as sqla_url
from sqlalchemy import exc as sa_exc
from sqlalchemy import inspect
from sqlalchemy import schema
from sqlalchemy import types
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_sqlite_setup_teardown(self):
    fixture = test_fixtures.AdHocDbFixture('sqlite:///foo.db')
    fixture.setUp()
    self.assertEqual(enginefacade._context_manager._factory._writer_engine.url, sqla_url.make_url('sqlite:///foo.db'))
    self.assertTrue(os.path.exists('foo.db'))
    fixture.cleanUp()
    self.assertFalse(os.path.exists('foo.db'))