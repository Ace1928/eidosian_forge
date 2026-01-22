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
@method('{')
def select_namespace_uri(self, context=None):
    if self[1].label.endswith('function'):
        yield self[1].evaluate(context)
        return
    elif context is None:
        raise self.missing_context()
    if isinstance(context, XPathSchemaContext):
        yield from self.select_xsd_nodes(context, self.value)
    elif self.xsd_types is None:
        for item in context.iter_children_or_self():
            if item.match_name(self.value):
                yield item
    else:
        for item in context.iter_children_or_self():
            if item.match_name(self.value):
                assert isinstance(item, (ElementNode, AttributeNode))
                if item.xsd_type is not None:
                    yield item
                else:
                    context.item = self.get_typed_node(item)
                    yield context.item