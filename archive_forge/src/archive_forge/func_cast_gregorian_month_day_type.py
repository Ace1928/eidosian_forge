from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('gMonthDay')
def cast_gregorian_month_day_type(self, value):
    if isinstance(value, GregorianMonthDay):
        return value
    try:
        if isinstance(value, UntypedAtomic):
            return GregorianMonthDay.fromstring(value.value)
        elif isinstance(value, (Date10, DateTime10)):
            return GregorianMonthDay(value.month, value.day, value.tzinfo)
        return GregorianMonthDay.fromstring(value)
    except ValueError as err:
        raise self.error('FORG0001', err)