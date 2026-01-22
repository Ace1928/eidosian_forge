import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def string_to_bytes(text, unit_system='IEC', return_int=False):
    """Converts a string into an float representation of bytes.

    The units supported for IEC / mixed::

        Kb(it), Kib(it), Mb(it), Mib(it), Gb(it), Gib(it), Tb(it), Tib(it),
        Pb(it), Pib(it), Eb(it), Eib(it), Zb(it), Zib(it), Yb(it), Yib(it),
        Rb(it), Rib(it), Qb(it), Qib(it)

        KB, KiB, MB, MiB, GB, GiB, TB, TiB, PB, PiB, EB, EiB, ZB, ZiB,
        YB, YiB, RB, RiB, QB, QiB

    The units supported for SI ::

        kb(it), Mb(it), Gb(it), Tb(it), Pb(it), Eb(it), Zb(it), Yb(it),
        Rb(it), Qb(it)

        kB, MB, GB, TB, PB, EB, ZB, YB, RB, QB

    SI units are interpreted as power-of-ten (e.g. 1kb = 1000b).  Note
    that the SI unit system does not support capital letter 'K'

    IEC units are interpreted as power-of-two (e.g. 1MiB = 1MB =
    1024b)

    Mixed units interpret the "i" to mean IEC, and no "i" to mean SI
    (e.g. 1kb = 1000b, 1kib == 1024b).  Additionaly, mixed units
    interpret 'K' as power-of-ten.  This mode is not particuarly
    useful for new code, but can help with compatability for parsers
    such as GNU parted.

    :param text: String input for bytes size conversion.
    :param unit_system: Unit system for byte size conversion.
    :param return_int: If True, returns integer representation of text
                       in bytes. (default: decimal)
    :returns: Numerical representation of text in bytes.
    :raises ValueError: If text has an invalid value.

    """
    try:
        base, reg_ex = UNIT_SYSTEM_INFO[unit_system]
    except KeyError:
        msg = _('Invalid unit system: "%s"') % unit_system
        raise ValueError(msg)
    match = reg_ex.match(text)
    if match:
        magnitude = float(match.group(1))
        unit_prefix = match.group(2)
        if match.group(3) in ['b', 'bit']:
            magnitude /= 8
        if unit_system == 'mixed':
            if unit_prefix and (not unit_prefix.endswith('i')):
                if unit_prefix.startswith == 'K':
                    unit_prefix = 'k'
                base = 1000
            else:
                base = 1024
    else:
        msg = _('Invalid string format: %s') % text
        raise ValueError(msg)
    if not unit_prefix:
        res = magnitude
    else:
        res = magnitude * pow(base, UNIT_PREFIX_EXPONENT[unit_prefix])
    if return_int:
        return int(math.ceil(res))
    return res