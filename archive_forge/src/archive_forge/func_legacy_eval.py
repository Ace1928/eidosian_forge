import testtools
import yaql
from yaql.language import factory
from yaql import legacy
def legacy_eval(self, expression, data=None, context=None):
    expr = self.legacy_engine(expression)
    return expr.evaluate(data=data, context=context or self.legacy_context)