from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('dayTimeDuration')
def cast_day_time_duration_type(self, value):
    if isinstance(value, DayTimeDuration):
        return value
    elif isinstance(value, Duration):
        return DayTimeDuration(seconds=value.seconds)
    try:
        if isinstance(value, UntypedAtomic):
            return DayTimeDuration.fromstring(value.value)
        return DayTimeDuration.fromstring(value)
    except OverflowError as err:
        raise self.error('FODT0002', err) from None
    except ValueError as err:
        raise self.error('FORG0001', err) from None