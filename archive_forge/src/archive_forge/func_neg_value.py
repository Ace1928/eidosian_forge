from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
@specs.yaql_property(int)
def neg_value(value):
    return -value