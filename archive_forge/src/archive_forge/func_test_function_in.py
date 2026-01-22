import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_function_in(self):

    def f():
        pass

    def f_():
        pass

    def f__():
        pass
    context = contexts.Context()
    context2 = context.create_child_context()
    context3 = context2.create_child_context()
    context.register_function(f)
    context.register_function(f_)
    context3.register_function(f__)
    functions = context3.collect_functions('f')
    self.assertNotIn(specs.get_function_definition(f__), context3)
    self.assertIn(functions[0].pop(), context3)
    self.assertNotIn(functions[1].pop(), context3)