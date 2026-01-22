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
@method('some', bp=20)
@method('every', bp=20)
def nud_quantified_expressions(self):
    del self[:]
    if self.parser.next_token.symbol != '$':
        return self.as_name()
    while True:
        self.parser.next_token.expected('$')
        variable = self.parser.expression(5)
        self.append(variable)
        self.parser.advance('in')
        expr = self.parser.expression(5)
        self.append(expr)
        for tk in filter(lambda x: x.symbol == '$', expr.iter()):
            if tk[0].value == variable[0].value:
                raise tk.error('XPST0008', 'loop variable in its range expression')
        if self.parser.next_token.symbol != ',':
            break
        self.parser.advance()
    self.parser.advance('satisfies')
    self.append(self.parser.expression(5))
    return self