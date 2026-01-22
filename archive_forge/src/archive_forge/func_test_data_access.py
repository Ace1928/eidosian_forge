from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_data_access(self):

    def foo(yaql_interface):
        return (yaql_interface[''], yaql_interface['key'])
    self.context.register_function(foo)
    self.context['key'] = 'value'
    self.assertEqual(['test', 'value'], self.eval('foo()', data='test'))