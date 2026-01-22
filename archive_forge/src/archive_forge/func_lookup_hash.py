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
def lookup_hash(digest, return_unknown=False, required=True):
    """
    Returns a :class:`HashInfo` record containing information about a given hash function.
    Can be used to look up a hash constructor by name, normalize hash name representation, etc.

    :arg digest:
        This can be any of:

        * A string containing a :mod:`!hashlib` digest name (e.g. ``"sha256"``),
        * A string containing an IANA-assigned hash name,
        * A digest constructor function (e.g. ``hashlib.sha256``).

        Case is ignored, underscores are converted to hyphens,
        and various other cleanups are made.

    :param required:
        By default (True), this function will throw an :exc:`~passlib.exc.UnknownHashError` if no hash constructor
        can be found, or if the hash is not actually available.

        If this flag is False, it will instead return a dummy :class:`!HashInfo` record
        which will defer throwing the error until it's constructor function is called.
        This is mainly used by :func:`norm_hash_name`.

    :param return_unknown:

        .. deprecated:: 1.7.3

            deprecated, and will be removed in passlib 2.0.
            this acts like inverse of **required**.

    :returns HashInfo:
        :class:`HashInfo` instance containing information about specified digest.

        Multiple calls resolving to the same hash should always
        return the same :class:`!HashInfo` instance.
    """
    cache = _hash_info_cache
    try:
        return cache[digest]
    except (KeyError, TypeError):
        pass
    if return_unknown:
        required = False
    cache_by_name = True
    if isinstance(digest, unicode_or_bytes_types):
        name_list = _get_hash_aliases(digest)
        name = name_list[0]
        assert name
        if name != digest:
            info = lookup_hash(name, required=required)
            cache[digest] = info
            return info
        const = _get_hash_const(name)
        if const and mock_fips_mode and (name not in _fips_algorithms):

            def const(source=b''):
                raise ValueError('%r disabled for fips by passlib set_mock_fips_mode()' % name)
    elif isinstance(digest, HashInfo):
        return digest
    elif callable(digest):
        const = digest
        name_list = _get_hash_aliases(const().name)
        name = name_list[0]
        other_const = _get_hash_const(name)
        if other_const is None:
            pass
        elif other_const is const:
            pass
        else:
            cache_by_name = False
    else:
        raise exc.ExpectedTypeError(digest, 'digest name or constructor', 'digest')
    info = HashInfo(const=const, names=name_list, required=required)
    if const is not None:
        cache[const] = info
    if cache_by_name:
        for name in name_list:
            if name:
                assert cache.get(name) in [None, info], '%r already in cache' % name
                cache[name] = info
    return info