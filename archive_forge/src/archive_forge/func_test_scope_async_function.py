from pyflakes import messages as m
from pyflakes.checker import (FunctionScope, ClassScope, ModuleScope,
from pyflakes.test.harness import TestCase
def test_scope_async_function(self):
    self.flakes('async def foo(): pass', is_segment=True)