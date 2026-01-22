from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('base64Binary')
@method('hexBinary')
def evaluate_binary_types(self, context=None):
    arg = self.data_value(self.get_argument(self.context or context))
    if arg is None:
        return []
    try:
        return self.cast(arg)
    except ElementPathError as err:
        if isinstance(context, XPathSchemaContext):
            return []
        err.token = self
        raise