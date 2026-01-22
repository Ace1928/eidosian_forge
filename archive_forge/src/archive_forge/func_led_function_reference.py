from copy import copy
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XSD_NAMESPACE
from ..xpath_nodes import AttributeNode, ElementNode
from ..xpath_tokens import XPathToken, ValueToken, XPathFunction, \
from ..xpath_context import XPathSchemaContext
from ..datatypes import QName
from .xpath30_parser import XPath30Parser
@method('#', bp=90)
def led_function_reference(self, left):
    if not left.label.endswith('function'):
        left.expected(':', '(name)', 'Q{')
    self[:] = (left, self.parser.expression(rbp=90))
    self[1].expected('(integer)')
    return self