import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_short_output(self) -> None:
    num_lines = 10
    with isolation.isolate(finite(num_lines)) as env:
        pager = isolation.PagerControl(env)
        for i, l in enumerate(pager.read_lines(num_lines)):
            self.assertEqual(str(i), l.rstrip())
        self.assertFalse(env.error_output())
    self.assertEqual(0, env.exit_code())