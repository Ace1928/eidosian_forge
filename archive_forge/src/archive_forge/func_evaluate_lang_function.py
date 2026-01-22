import math
import decimal
from ..helpers import get_double
from ..datatypes import Duration, DayTimeDuration, YearMonthDuration, \
from ..namespaces import XML_ID, XML_LANG, get_prefixed_name
from ..xpath_nodes import XPathNode, ElementNode, TextNode, CommentNode, \
from ..xpath_tokens import XPathFunction
from ..xpath_context import XPathSchemaContext
from ._xpath1_operators import XPath1Parser
@method(function('lang', nargs=1, sequence_types=('xs:string?', 'xs:boolean')))
def evaluate_lang_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    if not isinstance(context.item, ElementNode) or isinstance(context.item, SchemaElementNode):
        return False
    else:
        try:
            attr = context.item.elem.attrib[XML_LANG]
        except KeyError:
            for e in context.iter_ancestors():
                if isinstance(e, ElementNode) and XML_LANG in e.elem.attrib:
                    lang = e.elem.attrib[XML_LANG]
                    break
            else:
                return False
        else:
            if not isinstance(attr, str):
                return False
            lang = attr.strip()
        if '-' in lang:
            lang, _ = lang.split('-')
        return lang.lower() == self[0].evaluate().lower()