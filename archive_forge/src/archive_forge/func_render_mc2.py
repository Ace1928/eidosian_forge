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
def render_mc2(ident, salt, checksum, sep=u('$')):
    """format hash using 2-part modular crypt format; inverse of parse_mc2()

    returns native string with format :samp:`{ident}{salt}[${checksum}]`,
    such as used by md5_crypt.

    :arg ident: identifier prefix (unicode)
    :arg salt: encoded salt (unicode)
    :arg checksum: encoded checksum (unicode or None)
    :param sep: separator char (unicode, defaults to ``$``)

    :returns:
        config or hash (native str)
    """
    if checksum:
        parts = [ident, salt, sep, checksum]
    else:
        parts = [ident, salt]
    return uascii_to_str(join_unicode(parts))