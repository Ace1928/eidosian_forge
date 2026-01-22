import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_function_definition(self):

    def func(a, b, *args, **kwargs):
        return (a, b, args, kwargs)
    fd = specs.get_function_definition(func)
    self.assertEqual((1, 2, (5, 7), {'kw1': 'x', 'kw2': None}), fd(self.engine, self.context)(1, 2, 5, 7, kw1='x', kw2=None))
    self.assertEqual((1, 5, (), {}), fd(self.engine, self.context)(1, b=5))