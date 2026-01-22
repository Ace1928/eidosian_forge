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
@method('//')
@method('/')
def led_child_or_descendant_path(self, left):
    if left.symbol in ('/', '//', ':', '[', '$'):
        pass
    elif left.label not in self.parser.PATH_STEP_LABELS and left.symbol not in self.parser.PATH_STEP_SYMBOLS:
        raise self.wrong_syntax()
    if self.parser.next_token.label not in self.parser.PATH_STEP_LABELS:
        self.parser.expected_next(*self.parser.PATH_STEP_SYMBOLS)
    self[:] = (left, self.parser.expression(75))
    return self