import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_resetCallsSetUpCleanUp(self):
    calls = []

    class FixtureWithSetupOnly(fixtures.Fixture):

        def setUp(self):
            super(FixtureWithSetupOnly, self).setUp()
            calls.append('setUp')
            self.addCleanup(calls.append, 'cleanUp')
    fixture = FixtureWithSetupOnly()
    fixture.setUp()
    fixture.reset()
    fixture.cleanUp()
    self.assertEqual(['setUp', 'cleanUp', 'setUp', 'cleanUp'], calls)