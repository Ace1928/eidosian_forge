from ..exceptions import ElementPathError, ElementPathSyntaxError
from ..namespaces import XSD_NAMESPACE
from ..datatypes import xsd10_atomic_types, xsd11_atomic_types, GregorianDay, \
from ..xpath_context import XPathSchemaContext
from ._xpath2_functions import XPath2Parser
@constructor('dateTime', bp=90, label=('function', 'constructor function'), nargs=(1, 2), sequence_types=('xs:date?', 'xs:time?', 'xs:dateTime?'))
def cast_datetime_type(self, value):
    cls = DateTime if self.parser.xsd_version == '1.1' else DateTime10
    if isinstance(value, cls):
        return value
    try:
        if isinstance(value, UntypedAtomic):
            return cls.fromstring(value.value)
        elif isinstance(value, Date10):
            return cls(value.year, value.month, value.day, tzinfo=value.tzinfo)
        return cls.fromstring(value)
    except OverflowError as err:
        raise self.error('FODT0001', err) from None
    except ValueError as err:
        raise self.error('FORG0001', err) from None