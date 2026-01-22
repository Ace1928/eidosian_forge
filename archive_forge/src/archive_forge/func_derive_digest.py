import logging; log = logging.getLogger(__name__)
from passlib.utils import consteq, saslprep, to_native_str, splitcomma
from passlib.utils.binary import ab64_decode, ab64_encode
from passlib.utils.compat import bascii_to_str, iteritems, u, native_string_types
from passlib.crypto.digest import pbkdf2_hmac, norm_hash_name
import passlib.utils.handlers as uh
@classmethod
def derive_digest(cls, password, salt, rounds, alg):
    """helper to create SaltedPassword digest for SCRAM.

        This performs the step in the SCRAM protocol described as::

            SaltedPassword  := Hi(Normalize(password), salt, i)

        :type password: unicode or utf-8 bytes
        :arg password: password to run through digest

        :type salt: bytes
        :arg salt: raw salt data

        :type rounds: int
        :arg rounds: number of iterations.

        :type alg: str
        :arg alg: name of digest to use (e.g. ``"sha-1"``).

        :returns:
            raw bytes of ``SaltedPassword``
        """
    if isinstance(password, bytes):
        password = password.decode('utf-8')
    return pbkdf2_hmac(alg, saslprep(password), salt, rounds)