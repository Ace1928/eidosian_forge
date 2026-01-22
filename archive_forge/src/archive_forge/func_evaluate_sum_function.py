import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('sum', nargs=(1, 2), sequence_types=('xs:anyAtomicType*', 'xs:anyAtomicType?', 'xs:anyAtomicType?')))
def evaluate_sum_function(self, context=None):
    if self.context is not None:
        context = self.context
    xsd_version = self.parser.xsd_version
    try:
        values = [get_double(self.string_value(x), xsd_version) if isinstance(x, XPathNode) else x for x in self[0].iter_flatten(context)]
    except (TypeError, ValueError):
        if self.parser.version == '1.0':
            return math.nan
        elif isinstance(context, XPathSchemaContext):
            return []
        raise self.error('FORG0006') from None
    if not values:
        zero = 0 if len(self) == 1 else self.get_argument(context, index=1)
        return [] if zero is None else zero
    if all((isinstance(x, (decimal.Decimal, int)) for x in values)):
        return sum(values) if len(values) > 1 else values[0]
    elif all((isinstance(x, DayTimeDuration) for x in values)) or all((isinstance(x, YearMonthDuration) for x in values)):
        return sum(values[1:], start=values[0])
    elif any((isinstance(x, Duration) for x in values)):
        raise self.error('FORG0006', 'invalid sum of duration values')
    elif any((isinstance(x, (StringProxy, AnyURI)) for x in values)):
        raise self.error('FORG0006', 'cannot apply fn:sum() to string-based types')
    elif any((isinstance(x, float) and math.isnan(x) for x in values)):
        return math.nan
    elif all((isinstance(x, Float10) for x in values)):
        return sum(values)
    try:
        return sum((self.number_value(x) for x in values))
    except TypeError:
        if self.parser.version == '1.0':
            return math.nan
        elif isinstance(context, XPathSchemaContext):
            return []
        raise self.error('FORG0006') from None