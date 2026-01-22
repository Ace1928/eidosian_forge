import tempfile
from yaql.cli.cli_functions import load_data
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_bool_is_not_an_integer(self):

    @specs.parameter('arg', yaqltypes.Integer())
    def foo(arg):
        return arg
    self.context.register_function(foo)
    self.assertEqual(2, self.eval('foo(2)'))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo(true)')