import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('string-length', nargs=(0, 1), sequence_types=('xs:string?', 'xs:integer')))
def evaluate_string_length_function(self, context=None):
    if self.context is not None:
        context = self.context
    if self:
        return len(self.get_argument(context, default_to_context=True, default='', cls=str))
    elif context is None:
        raise self.missing_context()
    else:
        return len(self.string_value(context.item))