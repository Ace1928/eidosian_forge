from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_percent(number: float | decimal.Decimal | str, format: str | NumberPattern | None=None, locale: Locale | str | None=LC_NUMERIC, decimal_quantization: bool=True, group_separator: bool=True, *, numbering_system: Literal['default'] | str='latn') -> str:
    """Return formatted percent value for a specific locale.

    >>> format_percent(0.34, locale='en_US')
    u'34%'
    >>> format_percent(25.1234, locale='en_US')
    u'2,512%'
    >>> format_percent(25.1234, locale='sv_SE')
    u'2\\xa0512\\xa0%'
    >>> format_percent(25.1234, locale='ar_EG', numbering_system='default')
    u'2٬512%'

    The format pattern can also be specified explicitly:

    >>> format_percent(25.1234, u'#,##0‰', locale='en_US')
    u'25,123‰'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_percent(23.9876, locale='en_US')
    u'2,399%'
    >>> format_percent(23.9876, locale='en_US', decimal_quantization=False)
    u'2,398.76%'

    >>> format_percent(229291.1234, locale='pt_BR', group_separator=False)
    u'22929112%'

    >>> format_percent(229291.1234, locale='pt_BR', group_separator=True)
    u'22.929.112%'

    :param number: the percent number to format
    :param format:
    :param locale: the `Locale` object or locale identifier
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    :param group_separator: Boolean to switch group separator on/off in a locale's
                            number format.
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    if not format:
        format = locale.percent_formats[None]
    pattern = parse_pattern(format)
    return pattern.apply(number, locale, decimal_quantization=decimal_quantization, group_separator=group_separator, numbering_system=numbering_system)