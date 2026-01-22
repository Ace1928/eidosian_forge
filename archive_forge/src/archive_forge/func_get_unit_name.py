from __future__ import annotations
import decimal
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.numbers import LC_NUMERIC, format_decimal
def get_unit_name(measurement_unit: str, length: Literal['short', 'long', 'narrow']='long', locale: Locale | str | None=LC_NUMERIC) -> str | None:
    """
    Get the display name for a measurement unit in the given locale.

    >>> get_unit_name("radian", locale="en")
    'radians'

    Unknown units will raise exceptions:

    >>> get_unit_name("battery", locale="fi")
    Traceback (most recent call last):
        ...
    UnknownUnitError: battery/long is not a known unit/length in fi

    :param measurement_unit: the code of a measurement unit.
                             Known units can be found in the CLDR Unit Validity XML file:
                             https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml

    :param length: "short", "long" or "narrow"
    :param locale: the `Locale` object or locale identifier
    :return: The unit display name, or None.
    """
    locale = Locale.parse(locale)
    unit = _find_unit_pattern(measurement_unit, locale=locale)
    if not unit:
        raise UnknownUnitError(unit=measurement_unit, locale=locale)
    return locale.unit_display_names.get(unit, {}).get(length)