from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
def set_hash(self, user, realm=None, hash=_UNSET):
    """
        semi-private helper which allows writing a hash directly;
        adds user & realm if needed.

        If ``self.default_realm`` has been set, this may be called
        with the syntax ``set_hash(user, hash)``,
        otherwise it must be called with all three arguments:
        ``set_hash(user, realm, hash)``.

        .. warning::
            does not (currently) do any validation of the hash string

        .. versionadded:: 1.7
        """
    if hash is _UNSET:
        realm, hash = (None, realm)
    if PY3 and isinstance(hash, str):
        hash = hash.encode(self.encoding)
    key = self._encode_key(user, realm)
    existing = self._set_record(key, hash)
    self._autosave()
    return existing