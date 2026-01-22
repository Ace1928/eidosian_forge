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
@method(nullary('*'))
def select_wildcard(self, context=None):
    if self:
        item = self.evaluate(context)
        if item or not isinstance(item, list):
            if context is not None:
                context.item = item
            yield item
    elif context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        for item in context.iter_children_or_self():
            if item is not None:
                self.add_xsd_type(item)
                yield item
    elif self.xsd_types is None:
        for item in context.iter_children_or_self():
            if item is None:
                pass
            elif context.axis == 'attribute':
                if isinstance(item, AttributeNode):
                    yield item
            elif isinstance(item, ElementNode):
                yield item
    else:
        for item in context.iter_children_or_self():
            if context.item is not None and context.is_principal_node_kind():
                if isinstance(item, (ElementNode, AttributeNode)) and item.xsd_type is not None:
                    yield item
                else:
                    context.item = self.get_typed_node(item)
                    yield context.item