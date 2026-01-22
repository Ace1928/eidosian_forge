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
@method('$')
def evaluate_variable_reference(self, context=None):
    if context is None:
        raise self.missing_context()
    try:
        value = context.variables[self[0].value]
    except KeyError as err:
        raise self.error('XPST0008', 'unknown variable %r' % str(err)) from None
    else:
        return value if value is not None else []