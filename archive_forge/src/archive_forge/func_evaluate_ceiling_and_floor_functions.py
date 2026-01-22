import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('ceiling', nargs=1, sequence_types=('xs:numeric?', 'xs:numeric?')))
@method(function('floor', nargs=1, sequence_types=('xs:numeric?', 'xs:numeric?')))
def evaluate_ceiling_and_floor_functions(self, context=None):
    if self.context is not None:
        context = self.context
    arg = self.get_argument(context)
    if arg is None:
        return math.nan if self.parser.version == '1.0' else []
    elif isinstance(arg, XPathNode) or self.parser.compatibility_mode:
        arg = self.number_value(arg)
    try:
        if math.isnan(arg) or math.isinf(arg):
            return arg
        if self.symbol == 'floor':
            return type(arg)(math.floor(arg))
        else:
            return type(arg)(math.ceil(arg))
    except TypeError as err:
        if isinstance(context, XPathSchemaContext):
            return []
        elif isinstance(arg, str):
            raise self.error('XPTY0004', err) from None
        raise self.error('FORG0006', err) from None