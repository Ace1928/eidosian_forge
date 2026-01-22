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
@method('element')
def nud_element_kind_test(self):
    self.parser.advance('(')
    if self.parser.next_token.symbol != ')':
        self.parser.expected_next('(name)', ':', '*', message='a QName or a wildcard expected')
        self[0:] = (self.parser.expression(5),)
        if self.parser.next_token.symbol == ',':
            self.parser.advance(',')
            self.parser.expected_next('(name)', ':', message='a QName expected')
            self[1:] = (self.parser.expression(80),)
            if self.parser.next_token.symbol in ('*', '+', '?'):
                self[1].occurrence = self.parser.next_token.symbol
                self.parser.advance()
    self.parser.advance(')')
    self.value = None
    return self