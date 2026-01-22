import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test__setUp_fails(self):

    class Subclass(fixtures.Fixture):

        def _setUp(self):
            self.addDetail('log', text_content('stuff'))
            1 / 0
    f = Subclass()
    e = self.assertRaises(fixtures.MultipleExceptions, f.setUp)
    self.assertRaises(TypeError, f.cleanUp)
    self.assertIsInstance(e.args[0][1], ZeroDivisionError)
    self.assertIsInstance(e.args[1][1], fixtures.SetupError)
    self.assertEqual('stuff', e.args[1][1].args[0]['log'].as_text())