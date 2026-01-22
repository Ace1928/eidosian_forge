import unittest
from ..pass_manager import (
def test_this_before_that_pass_constraint(self) -> None:
    passes = [lambda x: 2 * x for _ in range(10)]
    pm = PassManager(passes)
    pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))
    self.assertRaises(RuntimeError, pm.validate)