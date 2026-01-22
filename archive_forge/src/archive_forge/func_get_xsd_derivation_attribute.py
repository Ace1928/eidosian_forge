from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def get_xsd_derivation_attribute(elem: Element, attribute: str, values: Optional[Set[str]]=None) -> str:
    """
    Get a derivation attribute (maybe 'block', 'blockDefault', 'final' or 'finalDefault')
    checking the items with the values arguments. Returns a string.

    :param elem: the Element instance.
    :param attribute: the attribute name.
    :param values: a set of admitted values when the attribute value is not '#all'.
    """
    value = elem.get(attribute)
    if value is None:
        return ''
    if values is None:
        values = XSD_FINAL_ATTRIBUTE_VALUES
    items = value.split()
    if len(items) == 1 and items[0] == '#all':
        return ' '.join(values)
    elif not all((s in values for s in items)):
        raise ValueError(_('wrong value %r for attribute %r') % (value, attribute))
    return value