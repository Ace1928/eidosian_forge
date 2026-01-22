from __future__ import division
import logging; log = logging.getLogger(__name__)
from passlib.exc import ExpectedTypeError
from passlib.utils.decor import deprecated_function
from passlib.utils.compat import native_string_types
from passlib.crypto.digest import norm_hash_name, lookup_hash, pbkdf1 as _pbkdf1, pbkdf2_hmac, compile_hmac
from warnings import warn
def get_prf(name):
    """Lookup pseudo-random family (PRF) by name.

    :arg name:
        This must be the name of a recognized prf.
        Currently this only recognizes names with the format
        :samp:`hmac-{digest}`, where :samp:`{digest}`
        is the name of a hash function such as
        ``md5``, ``sha256``, etc.

        todo: restore text about callables.

    :raises ValueError: if the name is not known
    :raises TypeError: if the name is not a callable or string

    :returns:
        a tuple of :samp:`({prf_func}, {digest_size})`, where:

        * :samp:`{prf_func}` is a function implementing
          the specified PRF, and has the signature
          ``prf_func(secret, message) -> digest``.

        * :samp:`{digest_size}` is an integer indicating
          the number of bytes the function returns.

    Usage example::

        >>> from passlib.utils.pbkdf2 import get_prf
        >>> hmac_sha256, dsize = get_prf("hmac-sha256")
        >>> hmac_sha256
        <function hmac_sha256 at 0x1e37c80>
        >>> dsize
        32
        >>> digest = hmac_sha256('password', 'message')

    .. deprecated:: 1.7

        This function is deprecated, and will be removed in Passlib 2.0.
        This only related replacement is :func:`passlib.crypto.digest.compile_hmac`.
    """
    global _prf_cache
    if name in _prf_cache:
        return _prf_cache[name]
    if isinstance(name, native_string_types):
        if not name.startswith(_HMAC_PREFIXES):
            raise ValueError('unknown prf algorithm: %r' % (name,))
        digest = lookup_hash(name[5:]).name

        def hmac(key, msg):
            return compile_hmac(digest, key)(msg)
        record = (hmac, hmac.digest_info.digest_size)
    elif callable(name):
        digest_size = len(name(b'x', b'y'))
        record = (name, digest_size)
    else:
        raise ExpectedTypeError(name, 'str or callable', 'prf name')
    _prf_cache[name] = record
    return record