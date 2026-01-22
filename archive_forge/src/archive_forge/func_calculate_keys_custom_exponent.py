import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def calculate_keys_custom_exponent(p, q, exponent):
    """Calculates an encryption and a decryption key given p, q and an exponent,
    and returns them as a tuple (e, d)

    :param p: the first large prime
    :param q: the second large prime
    :param exponent: the exponent for the key; only change this if you know
        what you're doing, as the exponent influences how difficult your
        private key can be cracked. A very common choice for e is 65537.
    :type exponent: int

    """
    phi_n = (p - 1) * (q - 1)
    try:
        d = rsa.common.inverse(exponent, phi_n)
    except rsa.common.NotRelativePrimeError as ex:
        raise rsa.common.NotRelativePrimeError(exponent, phi_n, ex.d, msg='e (%d) and phi_n (%d) are not relatively prime (divider=%i)' % (exponent, phi_n, ex.d))
    if exponent * d % phi_n != 1:
        raise ValueError('e (%d) and d (%d) are not mult. inv. modulo phi_n (%d)' % (exponent, d, phi_n))
    return (exponent, d)