import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_custom_cleanUp(self):

    class Stub:
        value = 42

        def mycleanup(self):
            self.value = None
    obj = Stub()
    fixture = fixtures.MethodFixture(obj, cleanup=obj.mycleanup)
    fixture.setUp()
    self.assertEqual(42, fixture.obj.value)
    self.assertEqual(obj, fixture.obj)
    fixture.cleanUp()
    self.assertEqual(None, fixture.obj.value)