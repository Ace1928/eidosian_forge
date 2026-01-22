import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_cleanup_only(self):

    class Stub:
        value = None

        def tearDown(self):
            self.value = 42
    fixture = fixtures.MethodFixture(Stub())
    fixture.setUp()
    self.assertEqual(None, fixture.obj.value)
    self.assertIsInstance(fixture.obj, Stub)
    fixture.cleanUp()
    self.assertEqual(42, fixture.obj.value)