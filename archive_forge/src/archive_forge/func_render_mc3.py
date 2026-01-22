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
def render_mc3(ident, rounds, salt, checksum, sep=u('$'), rounds_base=10):
    """format hash using 3-part modular crypt format; inverse of parse_mc3()

    returns native string with format :samp:`{ident}[{rounds}$]{salt}[${checksum}]`,
    such as used by sha1_crypt.

    :arg ident: identifier prefix (unicode)
    :arg rounds: rounds field (int or None)
    :arg salt: encoded salt (unicode)
    :arg checksum: encoded checksum (unicode or None)
    :param sep: separator char (unicode, defaults to ``$``)
    :param rounds_base: base to encode rounds value (defaults to base 10)

    :returns:
        config or hash (native str)
    """
    if rounds is None:
        rounds = u('')
    elif rounds_base == 16:
        rounds = u('%x') % rounds
    else:
        assert rounds_base == 10
        rounds = unicode(rounds)
    if checksum:
        parts = [ident, rounds, sep, salt, sep, checksum]
    else:
        parts = [ident, rounds, sep, salt]
    return uascii_to_str(join_unicode(parts))