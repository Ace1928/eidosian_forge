import math
import operator
from copy import copy
from decimal import Decimal, DivisionByZero
from ..exceptions import ElementPathError
from ..helpers import OCCURRENCE_INDICATORS, numeric_equal, numeric_not_equal, \
from ..namespaces import XSD_NAMESPACE, XSD_NOTATION, XSD_ANY_ATOMIC_TYPE, \
from ..datatypes import get_atomic_value, UntypedAtomic, QName, AnyURI, \
from ..xpath_nodes import ElementNode, DocumentNode, XPathNode, AttributeNode
from ..sequence_types import is_instance
from ..xpath_context import XPathSchemaContext
from ..xpath_tokens import XPathFunction
from .xpath2_parser import XPath2Parser
@method('attribute')
def nud_attribute_kind_test_or_axis(self):
    if self.parser.next_token.symbol == '::':
        self.label = 'axis'
        self.parser.advance('::')
        self.parser.expected_next('(name)', '*', 'text', 'node', 'document-node', 'comment', 'processing-instruction', 'attribute', 'schema-attribute', 'element', 'schema-element', 'namespace-node')
        self[:] = (self.parser.expression(rbp=90),)
    else:
        self.label = 'kind test'
        self.parser.advance('(')
        if self.parser.next_token.symbol != ')':
            self.parser.next_token.expected('(name)', '*', ':')
            self[:] = (self.parser.expression(5),)
            if self.parser.next_token.symbol == ',':
                self.parser.advance(',')
                self.parser.next_token.expected('(name)', ':')
                self[1:] = (self.parser.expression(5),)
        self.parser.advance(')')
        if self.namespace:
            msg = f'{self.value!r} is not allowed as function name'
            raise self.error('XPST0003', msg)
    return self