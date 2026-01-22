import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_appends_scenario_name_to_short_description(self):
    raw_test = self.ReferenceTest('test_pass_with_docstring')
    modified_test = apply_scenario(self.scenario, raw_test)
    raw_doc = self.ReferenceTest.test_pass_with_docstring.__doc__
    raw_desc = raw_doc.split('\n')[0].strip()
    scenario_name = self.scenario_name
    expect_desc = '%(raw_desc)s (%(scenario_name)s)' % vars()
    self.assertEqual(expect_desc, modified_test.shortDescription())