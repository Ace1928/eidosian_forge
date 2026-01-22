from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_method_call(self):

    def foo(yaql_interface):
        return yaql_interface.on([1, 2, 3]).where(lambda i: i > 1)

    @specs.inject('yi', yaqltypes.YaqlInterface())
    def bar(yi):
        return yi.on([1, 2, 3]).select(yi.engine('$ * $'))
    self.context.register_function(foo)
    self.context.register_function(bar)
    self.assertEqual([2, 3], self.eval('foo()'))
    self.assertEqual([1, 4, 9], self.eval('bar()'))