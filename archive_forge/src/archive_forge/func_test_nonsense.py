import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_nonsense(self):
    self.assertEvaled('!@#$ [1].a|bc', [1])
    self.assertEvaled('--- [2][0].a|bc', 2)
    self.assertCannotEval('"asdf".centered()[1].a|bc')
    self.assertEvaled('"asdf"[1].a|bc', 's')