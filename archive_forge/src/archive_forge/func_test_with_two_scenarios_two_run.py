import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
def test_with_two_scenarios_two_run(self):

    class ReferenceTest(self.Implementation):
        scenarios = [('1', {}), ('2', {})]

        def test_pass(self):
            pass
    test = ReferenceTest('test_pass')
    log = []
    result = LoggingResult(log)
    test.run(result)
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(2, result.testsRun)
    self.expectThat(log[0][1].id(), EndsWith('ReferenceTest.test_pass(1)'))
    self.expectThat(log[4][1].id(), EndsWith('ReferenceTest.test_pass(2)'))