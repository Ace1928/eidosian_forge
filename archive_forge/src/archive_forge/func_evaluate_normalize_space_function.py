import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('normalize-space', nargs=(0, 1), sequence_types=('xs:string?', 'xs:string')))
def evaluate_normalize_space_function(self, context=None):
    if self.context is not None:
        context = self.context
    if self.parser.version == '1.0' or not self:
        arg = self.string_value(self.get_argument(context, default_to_context=True, default=''))
    else:
        arg = self.get_argument(context, default_to_context=True, default='', cls=str)
    return ' '.join(arg.strip().split())