import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_boolean_constant_function(self):

    @specs.parameter('arg', yaqltypes.BooleanConstant())
    def foo(arg):
        return arg
    context = self.context.create_child_context()
    context.register_function(foo)
    self.assertTrue(self.eval('foo(true)', context=context))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', context=context, data=True)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo("true")', context=context)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo(null)', context=context)