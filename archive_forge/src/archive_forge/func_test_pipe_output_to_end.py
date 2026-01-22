import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_pipe_output_to_end(self) -> None:
    num_lines = 100
    with isolation.isolate(finite(num_lines), stdout_pipe=True) as env:
        with env.stdout_pipe() as out:
            output = out.readlines()
        self.assertEqual(num_lines, len(output))
        self.assertFalse(env.error_output())
    self.assertEqual(0, env.exit_code())