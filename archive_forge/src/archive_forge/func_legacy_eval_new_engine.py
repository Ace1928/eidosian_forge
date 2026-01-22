import testtools
import yaql
from yaql.language import factory
from yaql import legacy
def legacy_eval_new_engine(self, expression, data=None, context=None):
    expr = self.engine(expression)
    return expr.evaluate(data=data, context=context or self.legacy_context)