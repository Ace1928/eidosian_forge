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
@method('/')
def select_child_path(self, context=None):
    """
    Child path expression. Selects child:: axis as default (when bind to '*' or '(name)').
    """
    if context is None:
        raise self.missing_context()
    elif not self:
        if isinstance(context.root, DocumentNode):
            yield context.root
    elif len(self) == 1:
        if isinstance(context.document, DocumentNode):
            context.item = context.document
        elif context.root is None or isinstance(context.root.parent, ElementNode):
            return
        else:
            context.item = context.root
        yield from self[0].select(context)
    else:
        items = set()
        for _ in context.inner_focus_select(self[0]):
            if not isinstance(context.item, XPathNode):
                msg = f'Intermediate step contains an atomic value {context.item!r}'
                raise self.error('XPTY0019', msg)
            for result in self[1].select(context):
                if not isinstance(result, XPathNode):
                    yield result
                elif result in items:
                    pass
                elif isinstance(result, ElementNode):
                    if result.elem not in items:
                        items.add(result)
                        yield result
                else:
                    items.add(result)
                    yield result
                    if isinstance(context, XPathSchemaContext):
                        self[1].add_xsd_type(result)