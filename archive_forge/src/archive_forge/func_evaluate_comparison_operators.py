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
@method('=')
@method('!=')
@method('<')
@method('>')
@method('<=')
@method('>=')
def evaluate_comparison_operators(self, context=None):
    op = OPERATORS_MAP[self.symbol]
    try:
        return any((op(x1, x2) for x1, x2 in self.iter_comparison_data(context)))
    except (TypeError, ValueError) as err:
        if isinstance(context, XPathSchemaContext):
            return False
        elif isinstance(err, ElementPathTypeError):
            raise
        elif isinstance(err, TypeError):
            raise self.error('XPTY0004', err) from None
        else:
            raise self.error('FORG0001', err) from None