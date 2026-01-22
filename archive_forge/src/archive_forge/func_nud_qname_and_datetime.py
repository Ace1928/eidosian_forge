from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('QName')
@method('dateTime')
def nud_qname_and_datetime(self):
    try:
        self.parser.advance('(')
        self[0:] = (self.parser.expression(5),)
        if self.parser.next_token.symbol == ',':
            if self.label != 'function':
                raise self.error('XPST0017', 'unexpected 2nd argument')
            self.label = 'function'
            self.parser.advance(',')
            self[1:] = (self.parser.expression(5),)
        elif self.label != 'constructor function' or self.namespace != XSD_NAMESPACE:
            raise self.error('XPST0017', '2nd argument missing')
        else:
            self.label = 'constructor function'
            self.nargs = 1
        self.parser.advance(')')
    except SyntaxError:
        raise self.error('XPST0017') from None
    self.value = None
    return self