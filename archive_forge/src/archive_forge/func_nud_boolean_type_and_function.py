from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('boolean')
def nud_boolean_type_and_function(self):
    self.parser.advance('(')
    if self.parser.next_token.symbol == ')':
        msg = 'Too few arguments: expected at least 1 argument'
        raise self.error('XPST0017', msg)
    self[0:] = (self.parser.expression(5),)
    if self.parser.next_token.symbol == ',':
        msg = 'Too many arguments: expected at most 1 argument'
        raise self.error('XPST0017', msg)
    self.parser.advance(')')
    self.value = None
    return self