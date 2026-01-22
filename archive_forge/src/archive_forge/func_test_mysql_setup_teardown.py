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
def test_mysql_setup_teardown(self):
    try:
        mysql_backend = provision.Backend.backend_for_database_type('mysql')
    except exception.BackendNotAvailable:
        self.skipTest('mysql backend not available')
    mysql_backend.create_named_database('adhoc_test')
    self.addCleanup(mysql_backend.drop_named_database, 'adhoc_test')
    url = mysql_backend.provisioned_database_url('adhoc_test')
    fixture = test_fixtures.AdHocDbFixture(url)
    fixture.setUp()
    self.assertEqual(enginefacade._context_manager._factory._writer_engine.url, url)
    fixture.cleanUp()