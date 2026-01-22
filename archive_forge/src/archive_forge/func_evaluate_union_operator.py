from copy import copy
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XSD_NAMESPACE
from ..xpath_nodes import AttributeNode, ElementNode
from ..xpath_tokens import XPathToken, ValueToken, XPathFunction, \
from ..xpath_context import XPathSchemaContext
from ..datatypes import QName
from .xpath30_parser import XPath30Parser
@method(infix('||', bp=32))
def evaluate_union_operator(self, context=None):
    return self.string_value(self.get_argument(context)) + self.string_value(self.get_argument(context, index=1))