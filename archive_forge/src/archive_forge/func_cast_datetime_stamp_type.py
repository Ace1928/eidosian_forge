from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('dateTimeStamp')
def cast_datetime_stamp_type(self, value):
    if isinstance(value, DateTimeStamp):
        return value
    elif isinstance(value, DateTime10):
        value = str(value)
    try:
        if isinstance(value, UntypedAtomic):
            return DateTimeStamp.fromstring(value.value)
        elif isinstance(value, Date):
            return DateTimeStamp(value.year, value.month, value.day, tzinfo=value.tzinfo)
        return DateTimeStamp.fromstring(value)
    except ValueError as err:
        raise self.error('FORG0001', err) from None