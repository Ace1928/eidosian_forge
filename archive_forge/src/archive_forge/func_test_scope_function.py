from pyflakes import messages as m
from pyflakes.checker import (FunctionScope, ClassScope, ModuleScope,
from pyflakes.test.harness import TestCase
def test_scope_function(self):
    checker = self.flakes('\n        def foo(a, b=1, *d, **e):\n            def bar(f, g=1, *h, **i):\n                pass\n        ', is_segment=True)
    scopes = checker.deadScopes
    module_scopes = [scope for scope in scopes if scope.__class__ is ModuleScope]
    function_scopes = [scope for scope in scopes if scope.__class__ is FunctionScope]
    self.assertEqual(len(module_scopes), 0)
    self.assertEqual(len(function_scopes), 2)
    function_scope_foo = function_scopes[1]
    function_scope_bar = function_scopes[0]
    self.assertIsInstance(function_scope_foo, FunctionScope)
    self.assertIsInstance(function_scope_bar, FunctionScope)
    self.assertIn('a', function_scope_foo)
    self.assertIn('b', function_scope_foo)
    self.assertIn('d', function_scope_foo)
    self.assertIn('e', function_scope_foo)
    self.assertIn('bar', function_scope_foo)
    self.assertIn('f', function_scope_bar)
    self.assertIn('g', function_scope_bar)
    self.assertIn('h', function_scope_bar)
    self.assertIn('i', function_scope_bar)
    self.assertIsInstance(function_scope_foo['bar'], FunctionDefinition)
    self.assertIsInstance(function_scope_foo['a'], Argument)
    self.assertIsInstance(function_scope_foo['b'], Argument)
    self.assertIsInstance(function_scope_foo['d'], Argument)
    self.assertIsInstance(function_scope_foo['e'], Argument)
    self.assertIsInstance(function_scope_bar['f'], Argument)
    self.assertIsInstance(function_scope_bar['g'], Argument)
    self.assertIsInstance(function_scope_bar['h'], Argument)
    self.assertIsInstance(function_scope_bar['i'], Argument)