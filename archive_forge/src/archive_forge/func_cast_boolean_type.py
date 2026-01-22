from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('boolean', label=('function', 'constructor function'), sequence_types=('item()*', 'xs:boolean'))
def cast_boolean_type(self, value):
    try:
        return BooleanProxy(value)
    except ValueError as err:
        raise self.error('FORG0001', err) from None
    except TypeError as err:
        raise self.error('XPTY0004', err) from None