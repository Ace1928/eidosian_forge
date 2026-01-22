import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_sets_specified_id(self):
    raw_test = self.ReferenceTest('test_pass')
    raw_id = 'ReferenceTest.test_pass'
    scenario_name = self.scenario_name
    expect_id = '%(raw_id)s(%(scenario_name)s)' % vars()
    modified_test = apply_scenario(self.scenario, raw_test)
    self.expectThat(modified_test.id(), EndsWith(expect_id))