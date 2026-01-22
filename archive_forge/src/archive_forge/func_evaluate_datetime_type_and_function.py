from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@method('dateTime')
def evaluate_datetime_type_and_function(self, context=None):
    if self.context is not None:
        context = self.context
    if self.label == 'constructor function':
        arg = self.data_value(self.get_argument(context))
        if arg is None:
            return []
        try:
            return self.cast(arg)
        except (ValueError, TypeError) as err:
            if isinstance(context, XPathSchemaContext):
                return []
            elif isinstance(err, ValueError):
                raise self.error('FORG0001', err) from None
            else:
                raise self.error('FORG0006', err) from None
    else:
        dt = self.get_argument(context, cls=Date10)
        tm = self.get_argument(context, 1, cls=Time)
        if dt is None or tm is None:
            return []
        elif dt.tzinfo == tm.tzinfo or tm.tzinfo is None:
            tzinfo = dt.tzinfo
        elif dt.tzinfo is None:
            tzinfo = tm.tzinfo
        else:
            raise self.error('FORG0008')
        if self.parser.xsd_version == '1.1':
            return DateTime(dt.year, dt.month, dt.day, tm.hour, tm.minute, tm.second, tm.microsecond, tzinfo)
        return DateTime10(dt.year, dt.month, dt.day, tm.hour, tm.minute, tm.second, tm.microsecond, tzinfo)