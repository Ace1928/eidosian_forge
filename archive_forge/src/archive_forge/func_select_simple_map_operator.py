from copy import copy
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XSD_NAMESPACE
from ..xpath_nodes import AttributeNode, ElementNode
from ..xpath_tokens import XPathToken, ValueToken, XPathFunction, \
from ..xpath_context import XPathSchemaContext
from ..datatypes import QName
from .xpath30_parser import XPath30Parser
@method(infix('!', bp=72))
def select_simple_map_operator(self, context=None):
    if context is None:
        raise self.missing_context()
    for context.item in context.inner_focus_select(self[0]):
        for result in self[1].select(copy(context)):
            yield result
            if isinstance(context, XPathSchemaContext) and isinstance(result, (AttributeNode, ElementNode)):
                self[1].add_xsd_type(result)