import errno
import os
import stat
import sys
from subprocess import check_output
def property_value_to_bytes(value):
    """
    Return a byte string, which represents the given ``value`` in a way
    suitable as raw value of an udev property.

    If ``value`` is a boolean object, it is converted to ``'1'`` or ``'0'``,
    depending on whether ``value`` is ``True`` or ``False``.  If ``value`` is a
    byte string already, it is returned unchanged.  Anything else is simply
    converted to a unicode string, and then passed to
    :func:`ensure_byte_string`.
    """
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, bytes):
        return value
    return ensure_byte_string(str(value))