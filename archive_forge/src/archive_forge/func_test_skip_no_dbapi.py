import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
def test_skip_no_dbapi(self):

    class FakeDatabaseOpportunisticFixture(test_fixtures.OpportunisticDbFixture):
        DRIVER = 'postgresql'

    class SomeTest(test_fixtures.OpportunisticDBTestMixin, test_base.BaseTestCase):
        FIXTURE = FakeDatabaseOpportunisticFixture

        def runTest(self):
            pass
    st = SomeTest()
    with mock.patch('oslo_db.sqlalchemy.provision.Backend.backends_by_database_type', {'postgresql': provision.Backend('postgresql', 'postgresql://')}):
        st._database_resources = {}
        st._db_not_available = {}
        st._schema_resources = {}
        with mock.patch('sqlalchemy.create_engine', mock.Mock(side_effect=ImportError())):
            self.assertEqual([], st.resources)
            ex = self.assertRaises(self.skipException, st.setUp)
    self.assertEqual("Backend 'postgresql' is unavailable: No DBAPI installed", str(ex))