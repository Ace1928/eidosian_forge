import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_native_str, to_bytes, render_bytes, consteq
from passlib.utils.compat import unicode, str_to_uascii
import passlib.utils.handlers as uh
from passlib.crypto.digest import lookup_hash
def create_hex_hash(digest, module=__name__, django_name=None, required=True):
    """
    create hex-encoded unsalted hasher for specified digest algorithm.

    .. versionchanged:: 1.7.3
        If called with unknown/supported digest, won't throw error immediately,
        but instead return a dummy hasher that will throw error when called.

        set ``required=True`` to restore old behavior.
    """
    info = lookup_hash(digest, required=required)
    name = 'hex_' + info.name
    if not info.supported:
        info.digest_size = 0
    hasher = type(name, (HexDigestHash,), dict(name=name, __module__=module, _hash_func=staticmethod(info.const), checksum_size=info.digest_size * 2, __doc__='This class implements a plain hexadecimal %s hash, and follows the :ref:`password-hash-api`.\n\nIt supports no optional or contextual keywords.\n' % (info.name,)))
    if not info.supported:
        hasher.supported = False
    if django_name:
        hasher.django_name = django_name
    return hasher