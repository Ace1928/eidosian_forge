import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_calls_apply_scenario(self):
    self.addCleanup(setattr, testscenarios.scenarios, 'apply_scenario', apply_scenario)
    log = []

    def capture(scenario, test):
        log.append((scenario, test))
    testscenarios.scenarios.apply_scenario = capture
    scenarios = ['foo', 'bar']
    result = list(apply_scenarios(scenarios, 'test'))
    self.assertEqual([('foo', 'test'), ('bar', 'test')], log)