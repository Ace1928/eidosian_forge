from testtools import TestCase
from testtools.tests.helpers import (
def test_is_stack_hidden_consistent_true(self):
    hide_testtools_stack(True)
    self.assertEqual(True, is_stack_hidden())