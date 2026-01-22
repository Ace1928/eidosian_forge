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
@method(infix('idiv', bp=45))
def evaluate_idiv_operator(self, context=None):
    op1, op2 = self.get_operands(context)
    if op1 is None or op2 is None:
        raise self.error('XPST0005')
    try:
        if math.isinf(op1):
            raise self.error('FOAR0001' if op2 == 0 else 'FOAR0002')
        elif math.isnan(op1) or math.isnan(op2):
            raise self.error('FOAR0002')
    except TypeError as err:
        if isinstance(context, XPathSchemaContext):
            return UntypedAtomic('1')
        raise self.error('XPTY0004', err) from None
    try:
        result = op1 // op2
    except (ZeroDivisionError, DivisionByZero):
        if isinstance(context, XPathSchemaContext):
            return UntypedAtomic('1')
        raise self.error('FOAR0001') from None
    else:
        if result >= 0 or isinstance(op1, Decimal) or isinstance(op2, Decimal) or (abs(op1) == abs(op2)):
            return int(result)
        else:
            return int(result) + 1