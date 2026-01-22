import struct
from passlib import exc
from passlib.utils.compat import join_byte_values, byte_elem_value, \
def des_encrypt_block(key, input, salt=0, rounds=1):
    """encrypt single block of data using DES, operates on 8-byte strings.

    :arg key:
        DES key as 7 byte string, or 8 byte string with parity bits
        (parity bit values are ignored).

    :arg input:
        plaintext block to encrypt, as 8 byte string.

    :arg salt:
        Optional 24-bit integer used to mutate the base DES algorithm in a
        manner specific to :class:`~passlib.hash.des_crypt` and its variants.
        The default value ``0`` provides the normal (unsalted) DES behavior.
        The salt functions as follows:
        if the ``i``'th bit of ``salt`` is set,
        bits ``i`` and ``i+24`` are swapped in the DES E-box output.

    :arg rounds:
        Optional number of rounds of to apply the DES key schedule.
        the default (``rounds=1``) provides the normal DES behavior,
        but :class:`~passlib.hash.des_crypt` and its variants use
        alternate rounds values.

    :raises TypeError: if any of the provided args are of the wrong type.
    :raises ValueError:
        if any of the input blocks are the wrong size,
        or the salt/rounds values are out of range.

    :returns:
        resulting 8-byte ciphertext block.
    """
    if isinstance(key, bytes):
        if len(key) == 7:
            key = expand_des_key(key)
        elif len(key) != 8:
            raise ValueError('key must be 7 or 8 bytes')
        key = _unpack64(key)
    else:
        raise exc.ExpectedTypeError(key, 'bytes', 'key')
    if isinstance(input, bytes):
        if len(input) != 8:
            raise ValueError('input block must be 8 bytes')
        input = _unpack64(input)
    else:
        raise exc.ExpectedTypeError(input, 'bytes', 'input')
    result = des_encrypt_int_block(key, input, salt, rounds)
    return _pack64(result)