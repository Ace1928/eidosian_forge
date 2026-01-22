from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Util.py3compat import bchr, is_bytes
from Cryptodome.PublicKey.ECC import (EccKey,
def import_private_key(encoded):
    """Create a new Ed25519 or Ed448 private key object,
    starting from the key encoded as raw ``bytes``,
    in the format described in RFC8032.

    Args:
      encoded (bytes):
        The EdDSA private key to import.
        It must be 32 bytes for Ed25519, and 57 bytes for Ed448.

    Returns:
      :class:`Cryptodome.PublicKey.EccKey` : a new ECC key object.

    Raises:
      ValueError: when the given key cannot be parsed.
    """
    if len(encoded) == 32:
        curve_name = 'ed25519'
    elif len(encoded) == 57:
        curve_name = 'ed448'
    else:
        raise ValueError('Incorrect length. Only EdDSA private keys are supported.')
    return construct(seed=encoded, curve=curve_name)