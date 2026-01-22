import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_details_from_child_fixtures_are_returned(self):
    parent = fixtures.Fixture()
    with parent:
        child = fixtures.Fixture()
        parent.useFixture(child)
        child.addDetail('foo', 'content')
        self.assertEqual({'foo': 'content'}, parent.getDetails())
        del child._details['foo']
        self.assertEqual({}, parent.getDetails())
        child.addDetail('foo', 'content')
    self.assertRaises(TypeError, parent.getDetails)