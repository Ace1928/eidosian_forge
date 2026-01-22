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
@method('(')
def led_parenthesized_expression(self, left):
    if left.symbol == '(name)':
        if left.value in self.parser.RESERVED_FUNCTION_NAMES:
            msg = f'{left.value!r} is not allowed as function name'
            raise left.error('XPST0003', msg)
        else:
            raise left.error('XPST0017', 'unknown function {!r}'.format(left.value))
    elif left.symbol == ':' and left[1].symbol == '(name)':
        if left[1].namespace == XSD_NAMESPACE:
            msg = 'unknown constructor function {!r}'.format(left[1].value)
            raise left[1].error('XPST0017', msg)
        raise left.error('XPST0017', 'unknown function {!r}'.format(left.value))
    if self.parser.next_token.symbol != ')':
        self[:] = (left, self.parser.expression())
    else:
        self[:] = (left,)
    self.parser.advance(')')
    return self