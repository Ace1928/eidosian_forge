from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('dateTimeStamp')
def nud_datetime_stamp_type(self):
    if self.parser.xsd_version == '1.0':
        raise self.wrong_syntax('xs:dateTimeStamp is not recognized unless XSD 1.1 is enabled')
    try:
        self.parser.advance('(')
        self[0:] = (self.parser.expression(5),)
        if self.parser.next_token.symbol == ',':
            msg = 'Too many arguments: expected at most 1 argument'
            raise self.error('XPST0017', msg)
        self.parser.advance(')')
        self.value = None
    except SyntaxError as err:
        raise self.error('XPST0017', str(err)) from None
    return self