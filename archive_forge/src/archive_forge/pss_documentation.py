from Cryptodome.Util.py3compat import bchr, bord, iter_range
import Cryptodome.Util.number
from Cryptodome.Util.number import (ceil_div,
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
Check if the  PKCS#1 PSS signature over a message is valid.

        This function is also called ``RSASSA-PSS-VERIFY`` and
        it is specified in
        `section 8.1.2 of RFC8037 <https://tools.ietf.org/html/rfc8017#section-8.1.2>`_.

        :parameter msg_hash:
            The hash that was carried out over the message. This is an object
            belonging to the :mod:`Cryptodome.Hash` module.
        :type parameter: hash object

        :parameter signature:
            The signature that needs to be validated.
        :type signature: bytes

        :raise ValueError: if the signature is not valid.
        