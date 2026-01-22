import unittest
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures import TestWithFixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
@skipIf(gather_details is None, 'gather_details() is not available.')
def test_useFixture_details_not_captured_from_setUp(self):

    class SomethingBroke(Exception):
        pass

    class BrokenFixture(fixtures.Fixture):

        def setUp(self):
            super(BrokenFixture, self).setUp()
            self.addDetail('content', text_content('foobar'))
            raise SomethingBroke()
    broken_fixture = BrokenFixture()

    class NonDetailedTestCase(TestWithFixtures, unittest.TestCase):

        def setUp(self):
            super(NonDetailedTestCase, self).setUp()
            self.useFixture(broken_fixture)

        def test(self):
            pass
    non_detailed_test_case = NonDetailedTestCase('test')
    self.assertRaises(SomethingBroke, non_detailed_test_case.setUp)