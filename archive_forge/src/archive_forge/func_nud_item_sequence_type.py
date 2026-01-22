import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method('node')
def nud_item_sequence_type(self):
    XPathFunction.nud(self)
    if self.parser.next_token.symbol in ('*', '+', '?'):
        self.occurrence = self.parser.next_token.symbol
        self.parser.advance()
    return self