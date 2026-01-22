from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
def norm_hash_name(name, format='hashlib'):
    """Normalize hash function name (convenience wrapper for :func:`lookup_hash`).

    :arg name:
        Original hash function name.

        This name can be a Python :mod:`~hashlib` digest name,
        a SCRAM mechanism name, IANA assigned hash name, etc.
        Case is ignored, and underscores are converted to hyphens.

    :param format:
        Naming convention to normalize to.
        Possible values are:

        * ``"hashlib"`` (the default) - normalizes name to be compatible
          with Python's :mod:`!hashlib`.

        * ``"iana"`` - normalizes name to IANA-assigned hash function name.
          For hashes which IANA hasn't assigned a name for, this issues a warning,
          and then uses a heuristic to return a "best guess" name.

    :returns:
        Hash name, returned as native :class:`!str`.
    """
    info = lookup_hash(name, required=False)
    if info.unknown:
        warn('norm_hash_name(): ' + info.error_text, exc.PasslibRuntimeWarning)
    if format == 'hashlib':
        return info.name
    elif format == 'iana':
        return info.iana_name
    else:
        raise ValueError('unknown format: %r' % (format,))