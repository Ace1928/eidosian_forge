from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('string', label=('function', 'constructor function'), nargs=(0, 1), sequence_types=('item()?', 'xs:string'))
def cast_string_type(self, value):
    return self.string_value(value)