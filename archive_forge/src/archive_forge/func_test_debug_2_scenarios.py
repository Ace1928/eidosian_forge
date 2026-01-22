import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
def test_debug_2_scenarios(self):
    log = []

    class ReferenceTest(self.Implementation):
        scenarios = [('1', {'foo': 1, 'bar': 2}), ('2', {'foo': 2, 'bar': 4})]

        def test_check_foo(self):
            log.append(self)
    test = ReferenceTest('test_check_foo')
    test.debug()
    self.assertEqual(2, len(log))
    self.assertEqual(None, log[0].scenarios)
    self.assertEqual(None, log[1].scenarios)
    self.assertNotEqual(log[0].id(), log[1].id())