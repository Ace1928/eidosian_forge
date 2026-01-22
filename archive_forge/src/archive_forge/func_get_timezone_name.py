from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_timezone_name(dt_or_tzinfo: _DtOrTzinfo=None, width: Literal['long', 'short']='long', uncommon: bool=False, locale: Locale | str | None=LC_TIME, zone_variant: Literal['generic', 'daylight', 'standard'] | None=None, return_zone: bool=False) -> str:
    """Return the localized display name for the given timezone. The timezone
    may be specified using a ``datetime`` or `tzinfo` object.

    >>> from datetime import time
    >>> dt = time(15, 30, tzinfo=get_timezone('America/Los_Angeles'))
    >>> get_timezone_name(dt, locale='en_US')  # doctest: +SKIP
    u'Pacific Standard Time'
    >>> get_timezone_name(dt, locale='en_US', return_zone=True)
    'America/Los_Angeles'
    >>> get_timezone_name(dt, width='short', locale='en_US')  # doctest: +SKIP
    u'PST'

    If this function gets passed only a `tzinfo` object and no concrete
    `datetime`,  the returned display name is independent of daylight savings
    time. This can be used for example for selecting timezones, or to set the
    time of events that recur across DST changes:

    >>> tz = get_timezone('America/Los_Angeles')
    >>> get_timezone_name(tz, locale='en_US')
    u'Pacific Time'
    >>> get_timezone_name(tz, 'short', locale='en_US')
    u'PT'

    If no localized display name for the timezone is available, and the timezone
    is associated with a country that uses only a single timezone, the name of
    that country is returned, formatted according to the locale:

    >>> tz = get_timezone('Europe/Berlin')
    >>> get_timezone_name(tz, locale='de_DE')
    u'Mitteleurop\\xe4ische Zeit'
    >>> get_timezone_name(tz, locale='pt_BR')
    u'Hor\\xe1rio da Europa Central'

    On the other hand, if the country uses multiple timezones, the city is also
    included in the representation:

    >>> tz = get_timezone('America/St_Johns')
    >>> get_timezone_name(tz, locale='de_DE')
    u'Neufundland-Zeit'

    Note that short format is currently not supported for all timezones and
    all locales.  This is partially because not every timezone has a short
    code in every locale.  In that case it currently falls back to the long
    format.

    For more information see `LDML Appendix J: Time Zone Display Names
    <https://www.unicode.org/reports/tr35/#Time_Zone_Fallback>`_

    .. versionadded:: 0.9

    .. versionchanged:: 1.0
       Added `zone_variant` support.

    :param dt_or_tzinfo: the ``datetime`` or ``tzinfo`` object that determines
                         the timezone; if a ``tzinfo`` object is used, the
                         resulting display name will be generic, i.e.
                         independent of daylight savings time; if `None`, the
                         current date in UTC is assumed
    :param width: either "long" or "short"
    :param uncommon: deprecated and ignored
    :param zone_variant: defines the zone variation to return.  By default the
                           variation is defined from the datetime object
                           passed in.  If no datetime object is passed in, the
                           ``'generic'`` variation is assumed.  The following
                           values are valid: ``'generic'``, ``'daylight'`` and
                           ``'standard'``.
    :param locale: the `Locale` object, or a locale string
    :param return_zone: True or False. If true then function
                        returns long time zone ID
    """
    dt, tzinfo = _get_dt_and_tzinfo(dt_or_tzinfo)
    locale = Locale.parse(locale)
    zone = _get_tz_name(dt_or_tzinfo)
    if zone_variant is None:
        if dt is None:
            zone_variant = 'generic'
        else:
            dst = tzinfo.dst(dt)
            zone_variant = 'daylight' if dst else 'standard'
    elif zone_variant not in ('generic', 'standard', 'daylight'):
        raise ValueError('Invalid zone variation')
    zone = get_global('zone_aliases').get(zone, zone)
    if return_zone:
        return zone
    info = locale.time_zones.get(zone, {})
    if width in info and zone_variant in info[width]:
        return info[width][zone_variant]
    metazone = get_global('meta_zones').get(zone)
    if metazone:
        metazone_info = locale.meta_zones.get(metazone, {})
        if width in metazone_info:
            name = metazone_info[width].get(zone_variant)
            if width == 'short' and name == NO_INHERITANCE_MARKER:
                name = metazone_info.get('long', {}).get(zone_variant)
            if name:
                return name
    if dt is not None:
        return get_timezone_gmt(dt, width=width, locale=locale)
    return get_timezone_location(dt_or_tzinfo, locale=locale)