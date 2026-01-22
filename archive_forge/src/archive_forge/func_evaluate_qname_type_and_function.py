from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('QName')
def evaluate_qname_type_and_function(self, context=None):
    if self.context is not None:
        context = self.context
    if self.label == 'constructor function':
        arg = self.data_value(self.get_argument(context))
        return [] if arg is None else self.cast(arg)
    else:
        uri = self.get_argument(context)
        qname = self.get_argument(context, index=1)
        try:
            return QName(uri, qname)
        except (TypeError, ValueError) as err:
            if isinstance(context, XPathSchemaContext):
                return []
            elif isinstance(err, TypeError):
                raise self.error('XPTY0004', err)
            else:
                raise self.error('FOCA0002', err)