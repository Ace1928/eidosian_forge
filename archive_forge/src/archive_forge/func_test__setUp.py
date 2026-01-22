import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test__setUp(self):

    class Subclass(fixtures.Fixture):

        def _setUp(self):
            self.fred = 1
            self.addCleanup(setattr, self, 'fred', 2)
    with Subclass() as f:
        self.assertEqual(1, f.fred)
    self.assertEqual(2, f.fred)