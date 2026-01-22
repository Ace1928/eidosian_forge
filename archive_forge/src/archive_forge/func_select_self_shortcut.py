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
@method(nullary('.'))
def select_self_shortcut(self, context=None):
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        for item in context.iter_self():
            if isinstance(item, (AttributeNode, ElementNode)):
                if item.is_schema_node():
                    self.add_xsd_type(item)
                elif item is context.root:
                    for xsd_element in item:
                        self.add_xsd_type(xsd_element)
            yield item
    elif self.xsd_types is None:
        for item in context.iter_self():
            if item is not None:
                yield item
            elif isinstance(context.root, DocumentNode):
                yield context.root
    else:
        for item in context.iter_self():
            if item is not None:
                if isinstance(item, (ElementNode, AttributeNode)) and item.xsd_type is not None:
                    yield item
                else:
                    context.item = self.get_typed_node(item)
                    yield context.item
            elif isinstance(context.root, DocumentNode):
                yield context.root