import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_reset_raises_if_cleanup_raises(self):

    class FixtureWithSetupOnly(fixtures.Fixture):

        def do_raise(self):
            raise Exception('foo')

        def setUp(self):
            super(FixtureWithSetupOnly, self).setUp()
            self.addCleanup(self.do_raise)
    fixture = FixtureWithSetupOnly()
    fixture.setUp()
    exc = self.assertRaises(Exception, fixture.reset)
    self.assertEqual(('foo',), exc.args)