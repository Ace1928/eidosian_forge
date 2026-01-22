import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def newkeys(nbits, accurate=True, poolsize=1, exponent=DEFAULT_EXPONENT):
    """Generates public and private keys, and returns them as (pub, priv).

    The public key is also known as the 'encryption key', and is a
    :py:class:`rsa.PublicKey` object. The private key is also known as the
    'decryption key' and is a :py:class:`rsa.PrivateKey` object.

    :param nbits: the number of bits required to store ``n = p*q``.
    :param accurate: when True, ``n`` will have exactly the number of bits you
        asked for. However, this makes key generation much slower. When False,
        `n`` may have slightly less bits.
    :param poolsize: the number of processes to use to generate the prime
        numbers. If set to a number > 1, a parallel algorithm will be used.
        This requires Python 2.6 or newer.
    :param exponent: the exponent for the key; only change this if you know
        what you're doing, as the exponent influences how difficult your
        private key can be cracked. A very common choice for e is 65537.
    :type exponent: int

    :returns: a tuple (:py:class:`rsa.PublicKey`, :py:class:`rsa.PrivateKey`)

    The ``poolsize`` parameter was added in *Python-RSA 3.1* and requires
    Python 2.6 or newer.

    """
    if nbits < 16:
        raise ValueError('Key too small')
    if poolsize < 1:
        raise ValueError('Pool size (%i) should be >= 1' % poolsize)
    if poolsize > 1:
        from rsa import parallel
        import functools
        getprime_func = functools.partial(parallel.getprime, poolsize=poolsize)
    else:
        getprime_func = rsa.prime.getprime
    p, q, e, d = gen_keys(nbits, getprime_func, accurate=accurate, exponent=exponent)
    n = p * q
    return (PublicKey(n, e), PrivateKey(n, e, d, p, q))