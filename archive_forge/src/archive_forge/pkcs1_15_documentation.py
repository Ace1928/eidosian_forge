import Cryptodome.Util.number
from Cryptodome.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Cryptodome.Util.asn1 import DerSequence, DerNull, DerOctetString, DerObjectId
Check if the  PKCS#1 v1.5 signature over a message is valid.

        This function is also called ``RSASSA-PKCS1-V1_5-VERIFY`` and
        it is specified in
        `section 8.2.2 of RFC8037 <https://tools.ietf.org/html/rfc8017#page-37>`_.

        :parameter msg_hash:
            The hash that was carried out over the message. This is an object
            belonging to the :mod:`Cryptodome.Hash` module.
        :type parameter: hash object

        :parameter signature:
            The signature that needs to be validated.
        :type signature: byte string

        :raise ValueError: if the signature is not valid.
        