from copy import copy
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XSD_NAMESPACE
from ..xpath_nodes import AttributeNode, ElementNode
from ..xpath_tokens import XPathToken, ValueToken, XPathFunction, \
from ..xpath_context import XPathSchemaContext
from ..datatypes import QName
from .xpath30_parser import XPath30Parser
@method('#')
def evaluate_function_reference(self, context=None):
    arity = self[1].value
    if isinstance(self[0], XPathFunction):
        token_class = self[0].__class__
        namespace = self[0].namespace
        name = self[0].name
        if isinstance(name, QName):
            qname = name
        else:
            qname = QName(None, f'anonymous {self[0].label}'.replace(' ', '-'))
    else:
        if self[0].symbol == ':':
            qname = QName(self[0][1].namespace, self[0].value)
        elif self[0].symbol == 'Q{':
            qname = QName(self[0][0].value, self[0][1].value)
        elif self[0].value in self.parser.RESERVED_FUNCTION_NAMES:
            msg = f'{self[0].value!r} is not allowed as function name'
            raise self.error('XPST0003', msg)
        else:
            qname = QName(XPATH_FUNCTIONS_NAMESPACE, self[0].value)
        namespace = qname.namespace
        local_name = qname.local_name
        if namespace == XSD_NAMESPACE and arity != 1:
            raise self.error('XPST0017', f'unknown function {qname.qname}#{arity}')
        if namespace == XPATH_FUNCTIONS_NAMESPACE and local_name in ('QName', 'dateTime') and (arity == 1):
            raise self.error('XPST0017', f'unknown function {qname.qname}#{arity}')
        try:
            token_class = self.parser.symbol_table[qname.expanded_name]
        except KeyError:
            try:
                token_class = self.parser.symbol_table[local_name]
            except KeyError:
                msg = f'unknown function {qname.qname}#{arity}'
                raise self.error('XPST0017', msg) from None
        if token_class.symbol == 'function' or not token_class.label.endswith('function'):
            raise self.error('XPST0003')
    try:
        func = token_class(self.parser, nargs=arity)
    except TypeError:
        msg = f'unknown function {qname.qname}#{arity}'
        raise self.error('XPST0017', msg) from None
    else:
        if func.namespace is None:
            func.namespace = namespace
        elif func.namespace != namespace:
            raise self.error('XPST0017', f'unknown function {qname.qname}#{arity}')
        func.context = copy(context)
        return func