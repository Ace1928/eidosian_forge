import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('id', nargs=1, sequence_types=('xs:string*', 'element()*')))
def select_id_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    value = self[0].evaluate(context)
    item = context.item
    if item is None:
        item = context.root
    if isinstance(item, (ElementNode, DocumentNode)):
        for element in item.iter_descendants():
            if isinstance(element, ElementNode) and element.elem.get(XML_ID) == value:
                yield element