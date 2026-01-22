import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_delegate_factory(self):

    @specs.parameter('name', yaqltypes.String())
    @specs.inject('__df__', yaqltypes.Delegate())
    def call_func(__df__, name, *args, **kwargs):
        return __df__(name, *args, **kwargs)
    context = self.context.create_child_context()
    context.register_function(call_func)
    self.assertEqual([1, 2], self.eval('callFunc(list, 1, 2)', context=context))
    self.assertEqual(6, self.eval("callFunc('#operator_*', 2, 3)", context=context))