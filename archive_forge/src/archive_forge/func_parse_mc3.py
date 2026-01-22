from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
def parse_mc3(hash, prefix, sep=_UDOLLAR, rounds_base=10, default_rounds=None, handler=None):
    """parse hash using 3-part modular crypt format.

    this expects a hash of the format :samp:`{prefix}[{rounds}]${salt}[${checksum}]`,
    such as sha1_crypt, and parses it into rounds / salt / checksum portions.
    tries to convert the rounds to an integer,
    and throws error if it has zero-padding.

    :arg hash: the hash to parse (bytes or unicode)
    :arg prefix: the identifying prefix (unicode)
    :param sep: field separator (unicode, defaults to ``$``).
    :param rounds_base:
        the numeric base the rounds are encoded in (defaults to base 10).
    :param default_rounds:
        the default rounds value to return if the rounds field was omitted.
        if this is ``None`` (the default), the rounds field is *required*.
    :param handler: handler class to pass to error constructors.

    :returns:
        a ``(rounds : int, salt, chk | None)`` tuple.
    """
    hash = to_unicode(hash, 'ascii', 'hash')
    assert isinstance(prefix, unicode)
    if not hash.startswith(prefix):
        raise exc.InvalidHashError(handler)
    assert isinstance(sep, unicode)
    parts = hash[len(prefix):].split(sep)
    if len(parts) == 3:
        rounds, salt, chk = parts
    elif len(parts) == 2:
        rounds, salt = parts
        chk = None
    else:
        raise exc.MalformedHashError(handler)
    if rounds.startswith(_UZERO) and rounds != _UZERO:
        raise exc.ZeroPaddedRoundsError(handler)
    elif rounds:
        rounds = int(rounds, rounds_base)
    elif default_rounds is None:
        raise exc.MalformedHashError(handler, 'empty rounds field')
    else:
        rounds = default_rounds
    return (rounds, salt, chk or None)