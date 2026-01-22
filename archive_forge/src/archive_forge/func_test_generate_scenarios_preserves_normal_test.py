import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_generate_scenarios_preserves_normal_test(self):

    class ReferenceTest(unittest.TestCase):

        def test_pass(self):
            pass
    test = ReferenceTest('test_pass')
    log = self.hook_apply_scenarios()
    self.assertEqual([test], list(generate_scenarios(test)))
    self.assertEqual([], log)