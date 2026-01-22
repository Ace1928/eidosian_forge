import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_delete_function(self):

    def f():
        pass

    def f_():
        pass
    context = contexts.Context()
    context.register_function(f)
    context2 = context.create_child_context()
    context2.register_function(f_)
    functions, is_exclusive = context2.get_functions('f')
    spec = functions.pop()
    self.assertIn(spec, context2)
    context2.delete_function(spec)
    self.assertNotIn(spec, context2)
    functions, is_exclusive = context.get_functions('f')
    self.assertThat(functions, matchers.HasLength(1))