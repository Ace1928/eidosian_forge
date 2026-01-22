import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_high_precedence_operator_insertion(self):
    engine_factory = factory.YaqlFactory(allow_delegates=True)
    engine_factory.insert_operator(None, True, ':', factory.OperatorType.BINARY_LEFT_ASSOCIATIVE, True)
    engine = engine_factory.create()
    data = {'a': [1]}
    expr = engine('$.a[0]')
    self.assertEqual(1, expr.evaluate(context=self.context, data=data))