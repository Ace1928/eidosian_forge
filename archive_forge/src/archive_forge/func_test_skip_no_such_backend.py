import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
def test_skip_no_such_backend(self):

    class FakeDatabaseOpportunisticFixture(test_fixtures.OpportunisticDbFixture):
        DRIVER = 'postgresql+nosuchdbapi'

    class SomeTest(test_fixtures.OpportunisticDBTestMixin, test_base.BaseTestCase):
        FIXTURE = FakeDatabaseOpportunisticFixture

        def runTest(self):
            pass
    st = SomeTest()
    ex = self.assertRaises(self.skipException, st.setUp)
    self.assertEqual("Backend 'postgresql+nosuchdbapi' is unavailable: No such backend", str(ex))