from __future__ import annotations
import logging # isort:skip
from typing import (
from .. import colors, palettes
from ..util.strings import nice_join
def enumeration(*values: Any, case_sensitive: bool=True, quote: bool=False) -> Enumeration:
    """ Create an |Enumeration| object from a sequence of values.

    Call ``enumeration`` with a sequence of (unique) strings to create an
    Enumeration object:

    .. code-block:: python

        #: Specify the horizontal alignment for rendering text
        TextAlign = enumeration("left", "right", "center")

    Args:
        values (str) : string enumeration values, passed as positional arguments

            The order of arguments is the order of the enumeration, and the
            first element will be considered the default value when used
            to create |Enum| properties.

    Keyword Args:
        case_sensitive (bool, optional) :
            Whether validation should consider case or not (default: True)

        quote (bool, optional):
            Whether values should be quoted in the string representations
            (default: False)

    Raises:
        ValueError if values empty, if any value is not a string or not unique

    Returns:
        Enumeration

    """
    if len(values) == 1 and hasattr(values[0], '__args__'):
        values = get_args(values[0])
    if not (values and all((isinstance(value, str) and value for value in values))):
        raise ValueError(f'expected a non-empty sequence of strings, got {nice_join(values)}')
    if len(values) != len(set(values)):
        raise ValueError(f'enumeration items must be unique, got {nice_join(values)}')
    attrs: dict[str, Any] = {value: value for value in values}
    attrs.update({'_values': list(values), '_default': values[0], '_case_sensitive': case_sensitive, '_quote': quote})
    return type('Enumeration', (Enumeration,), attrs)()