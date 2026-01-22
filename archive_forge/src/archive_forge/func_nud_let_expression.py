from copy import copy
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XSD_NAMESPACE
from ..xpath_nodes import AttributeNode, ElementNode
from ..xpath_tokens import XPathToken, ValueToken, XPathFunction, \
from ..xpath_context import XPathSchemaContext
from ..datatypes import QName
from .xpath30_parser import XPath30Parser
@method(register('let', lbp=20, rbp=20, label='let expression'))
def nud_let_expression(self):
    del self[:]
    if self.parser.next_token.symbol != '$':
        return self.as_name()
    while True:
        self.parser.next_token.expected('$')
        variable = self.parser.expression(5)
        self.append(variable)
        self.parser.advance(':=')
        expr = self.parser.expression(5)
        self.append(expr)
        if self.parser.next_token.symbol != ',':
            break
        self.parser.advance()
    self.parser.advance('return')
    self.append(self.parser.expression(5))
    return self