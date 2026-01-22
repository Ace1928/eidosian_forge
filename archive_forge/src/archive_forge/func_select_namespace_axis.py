from ..xpath_nodes import ElementNode
from ..xpath_context import XPathSchemaContext
from ._xpath1_functions import XPath1Parser
@method(axis('namespace'))
def select_namespace_axis(self, context=None):
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        return
    elif isinstance(context.item, ElementNode):
        elem = context.item
        if self[0].symbol == 'namespace-node':
            name = '*'
        else:
            name = self[0].value
        for context.item in elem.namespace_nodes:
            if name == '*' or name == context.item.prefix:
                yield context.item