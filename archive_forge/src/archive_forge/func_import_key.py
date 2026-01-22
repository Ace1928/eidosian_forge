from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
def import_key(encoded, passphrase=None, curve_name=None):
    """Import an ECC key (public or private).

    Args:
      encoded (bytes or multi-line string):
        The ECC key to import.
        The function will try to automatically detect the right format.

        Supported formats for an ECC **public** key:

        * X.509 certificate: binary (DER) or ASCII (PEM).
        * X.509 ``subjectPublicKeyInfo``: binary (DER) or ASCII (PEM).
        * SEC1_ (or X9.62), as ``bytes``. NIST P curves only.
          You must also provide the ``curve_name`` (with a value from the `ECC table`_)
        * OpenSSH line, defined in RFC5656_ and RFC8709_ (ASCII).
          This is normally the content of files like ``~/.ssh/id_ecdsa.pub``.

        Supported formats for an ECC **private** key:

        * A binary ``ECPrivateKey`` structure, as defined in `RFC5915`_ (DER).
          NIST P curves only.
        * A `PKCS#8`_ structure (or the more recent Asymmetric Key Package, RFC5958_): binary (DER) or ASCII (PEM).
        * `OpenSSH 6.5`_ and newer versions (ASCII).

        Private keys can be in the clear or password-protected.

        For details about the PEM encoding, see `RFC1421`_/`RFC1423`_.

      passphrase (byte string):
        The passphrase to use for decrypting a private key.
        Encryption may be applied protected at the PEM level (not recommended)
        or at the PKCS#8 level (recommended).
        This parameter is ignored if the key in input is not encrypted.

      curve_name (string):
        For a SEC1 encoding only. This is the name of the curve,
        as defined in the `ECC table`_.

    .. note::

        To import EdDSA private and public keys, when encoded as raw ``bytes``, use:

        * :func:`Cryptodome.Signature.eddsa.import_public_key`, or
        * :func:`Cryptodome.Signature.eddsa.import_private_key`.

    Returns:
      :class:`EccKey` : a new ECC key object

    Raises:
      ValueError: when the given key cannot be parsed (possibly because
        the pass phrase is wrong).

    .. _RFC1421: https://datatracker.ietf.org/doc/html/rfc1421
    .. _RFC1423: https://datatracker.ietf.org/doc/html/rfc1423
    .. _RFC5915: https://datatracker.ietf.org/doc/html/rfc5915
    .. _RFC5656: https://datatracker.ietf.org/doc/html/rfc5656
    .. _RFC8709: https://datatracker.ietf.org/doc/html/rfc8709
    .. _RFC5958: https://datatracker.ietf.org/doc/html/rfc5958
    .. _`PKCS#8`: https://datatracker.ietf.org/doc/html/rfc5208
    .. _`OpenSSH 6.5`: https://flak.tedunangst.com/post/new-openssh-key-format-and-bcrypt-pbkdf
    .. _SEC1: https://www.secg.org/sec1-v2.pdf
    """
    from Cryptodome.IO import PEM
    encoded = tobytes(encoded)
    if passphrase is not None:
        passphrase = tobytes(passphrase)
    if encoded.startswith(b'-----BEGIN OPENSSH PRIVATE KEY'):
        text_encoded = tostr(encoded)
        openssh_encoded, marker, enc_flag = PEM.decode(text_encoded, passphrase)
        result = _import_openssh_private_ecc(openssh_encoded, passphrase)
        return result
    elif encoded.startswith(b'-----'):
        text_encoded = tostr(encoded)
        ecparams_start = '-----BEGIN EC PARAMETERS-----'
        ecparams_end = '-----END EC PARAMETERS-----'
        text_encoded = re.sub(ecparams_start + '.*?' + ecparams_end, '', text_encoded, flags=re.DOTALL)
        der_encoded, marker, enc_flag = PEM.decode(text_encoded, passphrase)
        if enc_flag:
            passphrase = None
        try:
            result = _import_der(der_encoded, passphrase)
        except UnsupportedEccFeature as uef:
            raise uef
        except ValueError:
            raise ValueError('Invalid DER encoding inside the PEM file')
        return result
    if encoded.startswith((b'ecdsa-sha2-', b'ssh-ed25519')):
        return _import_openssh_public(encoded)
    if len(encoded) > 0 and bord(encoded[0]) == 48:
        return _import_der(encoded, passphrase)
    if len(encoded) > 0 and bord(encoded[0]) in (2, 3, 4):
        if curve_name is None:
            raise ValueError('No curve name was provided')
        return _import_public_der(encoded, curve_name=curve_name)
    raise ValueError('ECC key format is not supported')