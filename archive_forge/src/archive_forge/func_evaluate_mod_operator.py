import math
import decimal
import operator
from copy import copy
from ..datatypes import AnyURI
from ..exceptions import ElementPathKeyError, ElementPathTypeError
from ..helpers import collapse_white_spaces, node_position
from ..datatypes import AbstractDateTime, Duration, DayTimeDuration, \
from ..xpath_context import XPathSchemaContext
from ..namespaces import XMLNS_NAMESPACE, XSD_NAMESPACE
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode
from ..xpath_tokens import XPathToken
from .xpath1_parser import XPath1Parser
@method(infix('mod', bp=45))
def evaluate_mod_operator(self, context=None):
    op1, op2 = self.get_operands(context, cls=NumericProxy)
    if op1 is None:
        return []
    elif op2 == 0 and isinstance(op2, float):
        return math.nan
    elif math.isinf(op2) and (not math.isinf(op1)) and (op1 != 0):
        return op1 if self.parser.version != '1.0' else math.nan
    try:
        if isinstance(op1, int) and isinstance(op2, int):
            return op1 % op2 if op1 * op2 >= 0 else -(abs(op1) % op2)
        return op1 % op2
    except TypeError as err:
        raise self.error('FORG0006', err) from None
    except (ZeroDivisionError, decimal.InvalidOperation):
        raise self.error('FOAR0001') from None