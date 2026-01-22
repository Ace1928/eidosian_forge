import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_preserves_scenarios_attribute(self):

    class ReferenceTest(unittest.TestCase):
        scenarios = [('demo', {})]

        def test_pass(self):
            pass
    test = ReferenceTest('test_pass')
    tests = list(apply_scenarios(ReferenceTest.scenarios, test))
    self.assertEqual([('demo', {})], ReferenceTest.scenarios)
    self.assertEqual(ReferenceTest.scenarios, tests[0].scenarios)